from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime

# Ensure required directories exist
if not os.path.isdir("static"):
    os.makedirs("static")
if not os.path.isdir("static/images"):
    os.makedirs("static/images")
if not os.path.isdir("static/css"):
    os.makedirs("static/css")

load_dotenv()

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Initialize models and OCR
model_path = os.getenv("MODEL_PATH")
yolo_model = YOLO(model_path)
paddle_ocr = PaddleOCR(use_angle_cls=False, lang="en")
DATABASE_URL = os.getenv("DATABASE_URL")

# Database setup
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Driver(Base):
    __tablename__ = "drivers"
    id = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String(50), unique=True, index=True)
    name = Column(String(100))
    vehicle_model = Column(String(100))
    registration_date = Column(Date)
    status = Column(String(100))
    image_filename = Column(String(200), nullable=True)

Base.metadata.create_all(bind=engine)

# Render the index (license plate recognition) page
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        if cls == 0 and conf > 0.8:
            license_plate_box = box.xyxy[0].cpu().numpy()
        elif cls == 1:
            license_plate_number_box = box.xyxy[0].cpu().numpy()

    if license_plate_box is None:
        return JSONResponse(
            status_code=404, 
            content={"detail": "License plate not detected with confidence > 80%.", "driver_found": False}
        )
    if license_plate_number_box is None:
        return JSONResponse(
            status_code=404, 
            content={"detail": "License plate number not detected.", "driver_found": False}
        )

    x1, y1, x2, y2 = map(int, license_plate_number_box)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return JSONResponse(
            status_code=400, 
            content={"detail": "Failed to crop license plate number region.", "driver_found": False}
        )

    ocr_results = paddle_ocr.ocr(cropped, det=False, rec=True)
    if not ocr_results or not ocr_results[0]:
        return JSONResponse(
            status_code=404, 
            content={"detail": "No text detected on the license plate number region.", "driver_found": False}
        )
    
    try:
        # Expecting the result format to be: [[(text, confidence)]]
        recognized_tuple = ocr_results[0][0]
        if isinstance(recognized_tuple, (list, tuple)) and len(recognized_tuple) == 2:
            license_text, confidence = recognized_tuple
        else:
            raise ValueError("Unexpected OCR result format")
    except Exception:
        return JSONResponse(
            status_code=404, 
            content={"detail": "Error processing OCR result.", "driver_found": False}
        )

    license_text = license_text.strip()

    session = SessionLocal()
    driver = session.query(Driver).filter(Driver.license_plate == license_text).first()
    session.close()

    if not driver:
        return JSONResponse(
            status_code=200, 
            content={
                "detail": "Driver's record not found for detected license plate.",
                "detected_license_plate": license_text,
                "driver_found": False,
                "detected": True
            }
        )

    driver_data = {
        "detail": "Driver's record found for detected license plate.",
        "id": driver.id,
        "license_plate": driver.license_plate,
        "name": driver.name,
        "image_url": f"/static/images/{driver.image_filename}" if driver.image_filename else None,
        "driver_found": True
    }
    return driver_data

# Render the driver management page
@app.get("/drivers", response_class=HTMLResponse)
async def drivers_page(request: Request):
    return templates.TemplateResponse("drivers.html", {"request": request})

@app.get("/get_drivers")
def get_drivers():
    session = SessionLocal()
    drivers = session.query(Driver).all()
    session.close()
    drivers_list = []
    for d in drivers:
        drivers_list.append({
            "id": d.id,
            "license_plate": d.license_plate,
            "name": d.name,
            "vehicle_model": d.vehicle_model,
            "registration_date": d.registration_date.strftime("%Y-%m-%d"),
            "status": d.status,
            "image_url": f"/static/images/{d.image_filename}" if d.image_filename else None
        })
    return drivers_list

# Create a new driver with an image upload
@app.post("/register_driver")
async def register_driver(
    license_plate: str = Form(...),
    name: str = Form(...),
    vehicle_model: str = Form(...),
    registration_date: str = Form(...),
    status: str = Form(...),
    image: UploadFile = File(...)
):
    session = SessionLocal()
    existing_driver = session.query(Driver).filter(Driver.license_plate == license_plate).first()
    if existing_driver:
        session.close()
        return JSONResponse(
            status_code=400, 
            content={"detail": "Driver record already exists for the provided license plate."}
        )

    # Save the uploaded image file
    images_dir = "static/images"
    os.makedirs(images_dir, exist_ok=True)
    file_extension = image.filename.split(".")[-1]
    new_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(images_dir, new_filename)
    with open(file_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Convert registration_date from string to date object (expecting format YYYY-MM-DD)
    try:
        registration_date_obj = datetime.strptime(registration_date, "%Y-%m-%d").date()
    except Exception:
        session.close()
        return JSONResponse(
            status_code=400, 
            content={"detail": "Invalid registration date format. Use YYYY-MM-DD."}
        )

    new_driver = Driver(
        license_plate=license_plate,
        name=name,
        vehicle_model=vehicle_model,
        registration_date=registration_date_obj,
        status=status,
        image_filename=new_filename
    )
    session.add(new_driver)
    session.commit()
    session.close()

    return {
        "success": True,
        "license_plate": license_plate,
        "name": name,
        "vehicle_model": vehicle_model,
        "registration_date": registration_date,
        "status": status,
        "image_url": f"/static/images/{new_filename}"
    }

# Update an existing driver. This endpoint now accepts all driver data.
# It requires 'old_license_plate' to identify the record and 'new_license_plate' as the updated value.
@app.post("/update_driver")
async def update_driver(
    id: int = Form(...),
    license_plate: str = Form(...),
    name: str = Form(...),
    vehicle_model: str = Form(...),
    registration_date: str = Form(...),
    status: str = Form(...),
    image: UploadFile = File(None)
):
    session = SessionLocal()
    driver = session.query(Driver).filter(Driver.id == id).first()
    if not driver:
        session.close()
        return JSONResponse(status_code=404, content={"detail": "Driver not found."})
    
    # Check if the license plate is being updated and ensure uniqueness
    if license_plate != driver.license_plate:
        if session.query(Driver).filter(Driver.license_plate == license_plate).first():
            session.close()
            return JSONResponse(
                status_code=400, 
                content={"detail": "Another driver with the new license plate already exists."}
            )
        driver.license_plate = license_plate

    driver.name = name
    driver.vehicle_model = vehicle_model
    try:
        registration_date_obj = datetime.strptime(registration_date, "%Y-%m-%d").date()
    except Exception:
        session.close()
        return JSONResponse(
            status_code=400, 
            content={"detail": "Invalid registration date format. Use YYYY-MM-DD."}
        )
    driver.registration_date = registration_date_obj
    driver.status = status

    if image is not None:
        # Optionally delete the old image file
        if driver.image_filename:
            old_file_path = os.path.join("static/images", driver.image_filename)
            if os.path.exists(old_file_path):
                os.remove(old_file_path)
        images_dir = "static/images"
        os.makedirs(images_dir, exist_ok=True)
        file_extension = image.filename.split(".")[-1]
        new_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(images_dir, new_filename)
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)
        driver.image_filename = new_filename

    session.commit()
    session.close()
    return {"detail": "Driver updated successfully", 'success': True}

@app.post("/delete_driver")
def delete_driver(id: int):
    session = SessionLocal()
    driver = session.query(Driver).filter(Driver.id == id).first()
    if driver:
        # Optionally delete the associated image file
        if driver.image_filename:
            file_path = os.path.join("static/images", driver.image_filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        session.delete(driver)
        session.commit()
    session.close()
    return {"detail": "Driver deleted successfully", "success": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
