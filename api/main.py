from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from ultralytics import YOLO
import shutil
import os
import cv2
import random

app = FastAPI(
    title="InfraSense AI",
    version="1.0.0"
)

# ============================
# API KEY
# Change this to your own secret key
# ============================
API_KEY = "InfraSenseAI2026"

# ============================
# Load trained model
# ============================
model = YOLO("runs/detect/runs/road_damage/weights/best.pt")


@app.get("/")
def home():
    return {
        "message": "InfraSense AI API Running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):

    # ============================
    # API Key Authentication
    # ============================
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    os.makedirs("temp", exist_ok=True)

    image_path = os.path.join("temp", file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ============================
    # Run Prediction
    # ============================
    results = model.predict(
        source=image_path,
        conf=0.10,
        save=False,
        save_txt=False
    )

    result = results[0]

    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    image_area = img_h * img_w

    class_names = model.names

    damage_report = []

    for box in result.boxes:

        cls = int(box.cls[0])

        x1, y1, x2, y2 = box.xyxy[0]

        width = x2 - x1
        height = y2 - y1

        area = width * height

        severity_score = (area / image_area) * 100

        if severity_score < 5:
            severity = "Low"
        elif severity_score < 15:
            severity = "Medium"
        elif severity_score < 30:
            severity = "High"
        else:
            severity = "Critical"

        priority_map = {
            "Low": "Low",
            "Medium": "Medium",
            "High": "High",
            "Critical": "Immediate"
        }

        damage_type = class_names[cls]

        if severity == "Low":
            repair_cost = random.randint(50000, 200000)
        elif severity == "Medium":
            repair_cost = random.randint(200000, 500000)
        elif severity == "High":
            repair_cost = random.randint(500000, 800000)
        else:
            repair_cost = random.randint(800000, 1000000)

        repair_time = {
            "Low": "2 Hours",
            "Medium": "1 Day",
            "High": "3 Days",
            "Critical": "7 Days"
        }

        display_confidence = round(float(box.conf[0]) * 100, 2)

        damage_report.append({
            "damage_type": damage_type,
            "confidence_score": display_confidence,
            "severity": severity,
            "priority": priority_map[severity],
            "estimated_repair_cost": repair_cost,
            "estimated_repair_time": repair_time[severity],
            "bounding_box": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]
        })

    # Delete uploaded image after prediction
    if os.path.exists(image_path):
        os.remove(image_path)

    return {
        "total_damages": len(damage_report),
        "detections": damage_report
    }