from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from ultralytics import YOLO
import shutil
import os
import cv2
import random

app = FastAPI(
    title="InfraSense AI",
    version="1.0.0",
    description="AI-based Road Damage Detection and Repair Recommendation API"
)


# ============================
# API KEY
# ============================
API_KEY = "InfraSenseAI2026"


# ============================
# Lazy Model Loading
# ============================
model = None


def get_model():
    global model

    if model is None:
        model_path = "weights/yolo12s_RDD2022_best.pt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        model = YOLO(model_path)

    return model



# ============================
# Home Route
# ============================
@app.get("/")
def home():
    return {
        "message": "InfraSense AI API Running",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }



# ============================
# Health Check
# ============================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "InfraSense AI"
    }



# ============================
# Prediction API
# ============================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):

    # API Authentication

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )


    os.makedirs("temp", exist_ok=True)

    image_path = os.path.join(
        "temp",
        file.filename
    )


    # Save uploaded image

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )


    # Load model

    model = get_model()


    # Run YOLO prediction

    results = model.predict(
        source=image_path,
        conf=0.10,
        save=False
    )


    result = results[0]


    image = cv2.imread(image_path)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )


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


        damage_percentage = (
            float(area) / image_area
        ) * 100



        # Severity Calculation

        if damage_percentage < 5:
            severity = "Low"

        elif damage_percentage < 15:
            severity = "Medium"

        elif damage_percentage < 30:
            severity = "High"

        else:
            severity = "Critical"



        priority_map = {

            "Low": "Low",

            "Medium": "Medium",

            "High": "High",

            "Critical": "Immediate"

        }



        # Cost Estimation

        if severity == "Low":
            repair_cost = random.randint(
                50000,
                200000
            )

        elif severity == "Medium":
            repair_cost = random.randint(
                200000,
                500000
            )

        elif severity == "High":
            repair_cost = random.randint(
                500000,
                800000
            )

        else:
            repair_cost = random.randint(
                800000,
                1000000
            )



        repair_time = {

            "Low": "2 Hours",

            "Medium": "1 Day",

            "High": "3 Days",

            "Critical": "7 Days"

        }



        confidence = round(
            float(box.conf[0]) * 100,
            2
        )


        damage_report.append({

            "damage_type":
                class_names[cls],

            "confidence_score":
                confidence,

            "severity":
                severity,

            "priority":
                priority_map[severity],

            "estimated_repair_cost":
                repair_cost,

            "estimated_repair_time":
                repair_time[severity],

            "bounding_box":[

                int(x1),
                int(y1),
                int(x2),
                int(y2)

            ]

        })


    # Delete temporary image

    if os.path.exists(image_path):

        os.remove(image_path)



    return {

        "total_damages":
            len(damage_report),

        "detections":
            damage_report

    }