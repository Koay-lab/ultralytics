"""Focused contract checks; fixture images and caches stay in temporary directories."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.data.augment import RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import verify_image_label
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import KeypointLoss


class OffimagePoseTests(unittest.TestCase):
    def test_verification_batching_and_cache_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            image = root / "images" / "sample.png"
            label = root / "labels" / "sample.txt"
            Image.new("RGB", (64, 64)).save(image)
            label.write_text("0 .5 .5 .8 .8 -.2 .5 1 1.2 .5 1 .4 .4 2 0 0 0\n")
            args = (str(image), str(label), "", True, 1, 4, 3, False)
            self.assertIsNotNone(verify_image_label(args, preserve_offimage_keypoints=True)[0])
            self.assertIsNone(verify_image_label(args)[0])
            data = {"names": {0: "animal"}, "kpt_shape": [4, 3], "preserve_offimage_keypoints": True}
            dataset = YOLODataset(str(root / "images"), data=data, task="pose", augment=False, imgsz=64)
            batch = YOLODataset.collate_fn([dataset[0]])
            np.testing.assert_allclose(batch["keypoints"][0, :, 0], [-0.2, 1.2, 0.4, 0], atol=1e-6)
            original_hash = dataset.get_cache_hash()
            dataset.data["preserve_offimage_keypoints"] = False
            self.assertNotEqual(dataset.get_cache_hash(), original_hash)
            for invalid in (
                "0 .5 .5 .8 .8 -.2 .5 2 1.2 .5 1 .4 .4 2 0 0 0",
                "0 .5 .5 .8 .8 -.2 .5 3 1.2 .5 1 .4 .4 2 0 0 0",
                "0 -.5 .5 .8 .8 -.2 .5 1 1.2 .5 1 .4 .4 2 0 0 0",
            ):
                label.write_text(invalid)
                self.assertIsNone(verify_image_label(args, preserve_offimage_keypoints=True)[0])

    def test_native_transforms_preserve_labels_and_coordinates(self):
        points = np.array([[[-10, 20, 1], [70, 30, 2], [20, 20, 0]]], np.float32)
        instance = Instances(
            np.array([[-10, -10, 80, 80]], np.float32),
            keypoints=points.copy(),
            bbox_format="xyxy",
            normalized=False,
            preserve_offimage_keypoints=True,
        )
        instance.clip(64, 64)
        np.testing.assert_array_equal(instance.keypoints[0, :, 2], [1, 1, 0])
        np.testing.assert_array_equal(instance.keypoints[0, :, 0], [-10, 70, 20])
        np.testing.assert_array_equal(instance.bboxes, [[0, 0, 64, 64]])
        self.assertTrue(Instances.concatenate([instance[:], instance[:]]).preserve_offimage_keypoints)
        transform = np.eye(3, dtype=np.float32)
        transform[0, 2] = -30
        transformed = RandomPerspective().apply_keypoints(points.copy(), transform, (64, 64), True)
        np.testing.assert_array_equal(transformed[0, :, 0], [-40, 40, -10])
        np.testing.assert_array_equal(transformed[0, :, 2], [1, 2, 0])
        ordinary = Instances(
            np.array([[-10, -10, 80, 80]], np.float32), keypoints=points.copy(), bbox_format="xyxy", normalized=False
        )
        ordinary.clip(64, 64)
        np.testing.assert_array_equal(ordinary.keypoints[0, :, 2], [0, 0, 0])

    def test_loss_and_validation_keep_offimage_supervision(self):
        target = torch.tensor([[[-10.0, 20.0, 1.0], [70.0, 30.0, 2.0], [0.0, 0.0, 0.0]]])
        prediction = (target + 1).requires_grad_()
        loss = KeypointLoss(torch.ones(3))(prediction, target, target[..., 2] != 0, torch.tensor([[100.0]]))
        loss.backward()
        self.assertTrue(torch.all(prediction.grad[0, :2, :2] != 0))
        self.assertTrue(torch.all(prediction.grad[0, 2] == 0))
        with tempfile.TemporaryDirectory() as temporary:
            validator = PoseValidator(save_dir=Path(temporary))
            validator.data = {"preserve_offimage_keypoints": True}
            pred = {"bboxes": torch.tensor([[0.0, 0.0, 64.0, 64.0]]), "keypoints": target}
            batch = {"imgsz": (64, 64), "ori_shape": (64, 64), "ratio_pad": ((1.0, 1.0), (0.0, 0.0))}
            result = validator.scale_preds(pred, batch)
            torch.testing.assert_close(result["keypoints"], target)
            validator.data = {}
            ordinary = validator.scale_preds(pred, batch)
            self.assertEqual(ordinary["keypoints"][0, 0, 0], 0)


if __name__ == "__main__":
    unittest.main()
