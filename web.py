from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

model_path = os.getenv("MODEL_PATH")
yolo_model = YOLO(model_path)
paddle_ocr = PaddleOCR(use_angle_cls=False, lang="en")
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Driver(Base):
    __tablename__ = "drivers"
    id = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String(50), unique=True, index=True)
    name = Column(String(100))

Base.metadata.create_all(bind=engine)

@app.post("/upload")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    results = yolo_model.predict(source=image, verbose=False)
    result = results[0]

    license_plate_box = None
    license_plate_number_box = None

    for box in result.boxes:
        cls = int(box.cls[0].item())
        conf = box.conf[0].item() if hasattr(box, "conf") else 0.0
        print(box)
        if cls == 0 and conf > 0.8:
            license_plate_box = box.xyxy[0].cpu().numpy()
        elif cls == 1:
            license_plate_number_box = box.xyxy[0].cpu().numpy()

    if license_plate_box is None:
        return JSONResponse(
            status_code=404, 
            content={"detail": "License plate not detected with confidence > 80%."}
        )
    if license_plate_number_box is None:
        return JSONResponse(
            status_code=404, 
            content={"detail": "License plate number not detected."}
        )

    x1, y1, x2, y2 = map(int, license_plate_number_box)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return JSONResponse(
            status_code=400, 
            content={"detail": "Failed to crop license plate number region."}
        )

    ocr_results = paddle_ocr.ocr(cropped, det=False, rec=True)

    if not ocr_results or not ocr_results[0]:
        return JSONResponse(
            status_code=404, 
            content={"detail": "No text detected on the license plate number region."}
        )
    
    first_line = ocr_results[0][0]  # (text, confidence) 
    if isinstance(first_line, tuple) and len(first_line) == 2:
        license_text, confidence = first_line
    else:
        license_text, confidence = first_line[1]

    license_text = license_text.strip()

    session = SessionLocal()
    driver = session.query(Driver).filter(Driver.license_plate == license_text).first()
    session.close()

    if not driver:
        return JSONResponse(
            status_code=404, 
            content={
                "detail": "Driver's record not found for detected license plate.",
                "detected_license_plate": license_text
            }
        )

    driver_data = {
        "id": driver.id,
        "license_plate": driver.license_plate,
        "name": driver.name,
    }
    return driver_data

@app.post("/register_driver")
async def register_driver(license_plate: str, name: str):
    session = SessionLocal()
    driver = session.query(Driver).filter(Driver.license_plate == license_plate).first()
    if driver:
        return JSONResponse(
            status_code=400, 
            content={"detail": "Driver record already exists for the provided license plate."}
        )

    new_driver = Driver(license_plate=license_plate, name=name)
    session.add(new_driver)
    session.commit()
    session.close()

    return {"license_plate": license_plate, "name": name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)