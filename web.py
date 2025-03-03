from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def read_index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>License Plate Recognition System</title>
      <style>
        /* Gradient background for the body */
        body {
          margin: 0;
          padding: 0;
          background: linear-gradient(135deg, #667eea, #764ba2);
          font-family: Arial, sans-serif;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          background-repeat: no-repeat;
          background-size: cover;
        }
        /* Glassmorphism container */
        .container {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 10px;
          padding: 20px;
          width: 90%;
          max-width: 600px;
          backdrop-filter: blur(10px);
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          text-align: center;
          color: #fff;
        }
        h1 {
          margin-bottom: 20px;
        }
        input[type="file"] {
          display: block;
          margin: 20px auto;
        }
        button {
          padding: 10px 20px;
          border: none;
          background-color: #007BFF;
          color: #fff;
          border-radius: 4px;
          cursor: pointer;
          margin-top: 10px;
        }
        button:hover {
          background-color: #0056b3;
        }
        #preview {
          margin-top: 20px;
          max-width: 100%;
          border-radius: 10px;
          display: none;
        }
        #result {
          margin-top: 20px;
          color: #fff;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>License Plate Recognition System</h1>
        <form id="uploadForm">
          <input type="file" id="fileInput" name="file" accept="image/*" required>
          <img id="preview" alt="License Plate Preview" />
          <button type="submit">Upload and Process</button>
        </form>
        <div id="result"></div>
      </div>
      <script>
        // Show a preview of the uploaded image immediately
        const fileInput = document.getElementById("fileInput");
        const preview = document.getElementById("preview");
        fileInput.addEventListener("change", function(){
            const file = this.files[0];
            if(file){
                const reader = new FileReader();
                reader.onload = function(e){
                    preview.src = e.target.result;
                    preview.style.display = "block";
                }
                reader.readAsDataURL(file);
            } else {
                preview.style.display = "none";
            }
        });

        // Submit the form and send the file to the /upload endpoint
        const form = document.getElementById('uploadForm');
        const resultDiv = document.getElementById('result');
    
        form.addEventListener('submit', async (e) => {
          e.preventDefault();
          const file = fileInput.files[0];
          if(!file){
            resultDiv.innerHTML = '<p style="color: red;">Please select an image file.</p>';
            return;
          }
          const formData = new FormData();
          formData.append('file', file);
    
          try {
            resultDiv.innerHTML = '<p>Processing image, please wait...</p>';
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok) {
              resultDiv.innerHTML = `<h2>Driver Information</h2>
                <p><strong>ID:</strong> ${data.id}</p>
                <p><strong>License Plate:</strong> ${data.license_plate}</p>
                <p><strong>Name:</strong> ${data.name}</p>`;
            } else {
              if(data.detected_license_plate != undefined || data.detected_license_plate != null){
                resultDiv.innerHTML += `<p>License Plate: ${data.detected_license_plate}</p>`
              }
              resultDiv.innerHTML += `<p style="color: red;">Error: ${data.detail}</p>`;
            }
          } catch (error) {
            resultDiv.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
          }
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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