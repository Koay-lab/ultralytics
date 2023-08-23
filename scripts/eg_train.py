from ultralytics import YOLO
from ultralytics import settings

settings.update({
    "runs_dir": "D:/Data/081823_masks_on/runs",
    "tensorboard": True,
})

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n-pose.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data="D:/Data/081823_masks_on/081823_masks_on.yaml",
                      epochs=10, imgsz=640, batch=-1,
                      save=True, save_period=10,
                      verbose=True,
                      cache=True, workers=32, device=[0, 1],
                      augment=True,
                      degrees=90,  # image translation (+/- fraction)
                      translate=0.1,  # image translation (+/- fraction)
                      scale=0.1,  # image scale (+/- gain)
                      perspective=0.1,  # image perspective (+/- fraction), range 0-0.001
                      flipud=0.7,  # image flip up-down (probability)
                      fliplr=0.5,  # image flip left-right (probability)
                      )

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("D:/Data/081823_masks_on/images/val/081823_masks_on_20230818-1353_cam21492306_frame600.png")

# Export the model to ONNX format
success = model.export(format='onnx')
