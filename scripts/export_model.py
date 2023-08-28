import torch
from ultralytics import YOLO


def main(model_file, gpu_id=1):
    model = YOLO(model_file)
    # model.val(device=gpu_id, workers=1)

    model.export(format="engine", device=gpu_id, imgsz=640, half=False, simplify=True)


if __name__ == "__main__":
    # main("D:/Data/081823_masks_on_spine/runs/pose/train/weights/epoch100.pt")
    # main("D:/Data/081823_masks_on_high_quality/runs/pose/train13/weights/best.pt")
    main("D:/Data/081823_masks_on_high_quality/runs/pose/train13/weights/epoch150.pt")
