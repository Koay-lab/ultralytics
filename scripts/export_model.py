import torch
from ultralytics import YOLO


def main(model_file, gpu_id=1):
    print(f"{torch.cuda.is_available()}: {torch.cuda.device_count()}")

    model = YOLO(model_file)
    model.val(device=gpu_id, workers=1)

    model.export(format="onnx", imgsz=640, simplify=True)
    # print(f"{torch.cuda.is_available()}: {torch.cuda.device_count()}")
    # model.export(format="engine", device="cuda", imgsz=640)


if __name__ == "__main__":
    # main("D:/Data/081823_masks_on_spine/runs/pose/train/weights/epoch100.pt")
    # main("D:/Data/081823_masks_on_high_quality/runs/pose/train13/weights/best.pt")
    # main("D:/Data/081823_masks_on_high_quality/runs/pose/train13/weights/epoch150.pt")
    # main("D:/Data/081823_masks_on_high_quality/runs/pose/train13/weights/epoch150.pt")
    # main("D:/Data/102723_reward_field_plus_before/runs/pose/train/weights/epoch50.pt")
    # main("D:/Data/102723_reward_field_plus_before/runs/pose/train/weights/epoch100.pt")
    main("D:/Data/102723_reward_field_edge_cases/runs/pose/train/weights/best.pt")
