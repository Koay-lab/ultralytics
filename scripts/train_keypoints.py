"""
Useful info:

** How keypoint visibility is treated by Yolo
https://github.com/ultralytics/ultralytics/issues/4199

* Coordinates must be clipped between [0, 1] even for invisible keypoints that are outside of the image, otherwise the
image and its annotation will be ignored as invalid.
"""


def main(data_spec):
    import os
    from ultralytics import YOLO
    from ultralytics import settings

    settings.update({
        "runs_dir": os.path.join(os.path.dirname(data_spec), "runs"),
        "tensorboard": True,
    })

    # Create a new YOLO model from scratch
    model = YOLO("yolov8n-pose-grayscale.yaml")
    model.load("yolov8n-pose.pt")

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('yolov8n-pose.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data=data_spec,
                          epochs=1000, imgsz=640, batch=64,
                          optimizer="auto",
                          # optimizer="AdamW", lr0=0.001,
                          save=True, save_period=50,
                          cache="disk", workers=8,
                          verbose=True, device=0,
                          augment=False,
                          )
    print(results)

    # Evaluate the model's performance on the validation set
    # results = model.val(device=0)
    # print(results)

    # Perform object detection on an image using the model
    # results = model("D:/Data/081823_masks_on/images/val/081823_masks_on_20230818-1353_cam21492306_frame600.png")
    # print(results)

    # Export the model to ONNX format
    # success = model.export(format='onnx')


if __name__ == "__main__":
    # freeze_support()
    # main("D:/Data/081823_masks_on_spine/081823_masks_on_spine.yaml")
    # main("D:/Data/081823_masks_on_high_quality/081823_masks_on_high_quality.yaml")
    # main("D:/Data/102723_reward_field_plus_before/102723_reward_field_plus_before.yaml")
    main("D:/Data/102723_reward_field_edge_cases/102723_reward_field_edge_cases.yaml")
