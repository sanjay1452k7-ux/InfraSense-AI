from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("Starting training...")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs",
    name="road_damage",
    device="cpu"
)

print("Training Finished!")