from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import uvicorn
import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
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


video_capture = None  # Camera starts only when needed
video_stream_active = False  # Flag to track active streaming
latest_detected_plate = None  # Ensure variable is always initialized

async def generate_frames():
    global video_capture, video_stream_active, latest_detected_plate

    if video_capture is None:
        video_capture = cv2.VideoCapture(0)

    video_stream_active = True  # Mark stream as active

    try:
        while video_stream_active:
            success, frame = video_capture.read()
            if not success:
                break

            # Perform license plate detection with YOLO
            results = yolo_model.predict(source=frame, verbose=False)
            result = results[0]

            for box in result.boxes:
                cls = int(box.cls[0].item())
                conf = box.conf[0].item() if hasattr(box, "conf") else 0.0

                if cls == 1 and conf > 0.8:  # If class is a license plate with high confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cropped = frame[y1:y2, x1:x2]  # Crop the detected license plate
                    if cropped.size > 0:
                        ocr_results = paddle_ocr.ocr(cropped, det=False, rec=True)

                        if ocr_results and ocr_results[0]:
                            license_text, confidence = ocr_results[0][0]
                            latest_detected_plate = license_text  # Save detected plate
                            cv2.putText(frame, license_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode and yield frame
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            await asyncio.sleep(0.05)  # Prevent CPU overload

    finally:
        video_stream_active = False
        if video_capture is not None:
            video_capture.release()
            video_capture = None

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

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

@app.post("/detect_license_plate")
def detect_license_plate():
    success, frame = video_capture.read()
    if not success:
        return JSONResponse(status_code=500, content={"detail": "Failed to capture frame."})
    
    results = yolo_model.predict(source=frame, verbose=False)
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
        return JSONResponse(status_code=404, content={"detail": "License plate not detected."})
    if license_plate_number_box is None:
        return JSONResponse(status_code=404, content={"detail": "License plate number not detected."})

    x1, y1, x2, y2 = map(int, license_plate_number_box)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return JSONResponse(status_code=400, content={"detail": "Failed to crop license plate."})

    ocr_results = paddle_ocr.ocr(cropped, det=False, rec=True)
    if not ocr_results or not ocr_results[0]:
        return JSONResponse(status_code=404, content={"detail": "No text detected."})
    
    license_text, confidence = ocr_results[0][0]
    return {
        "license_plate": license_text,
        "bounding_box": {
            "x1": int(license_plate_number_box[0]),
            "y1": int(license_plate_number_box[1]),
            "x2": int(license_plate_number_box[2]),
            "y2": int(license_plate_number_box[3])
        }
    }

@app.get("/stop_video")
async def stop_video():
    """Stops the video stream when the page is closed"""
    global video_stream_active, video_capture

    video_stream_active = False

    if video_capture is not None:
        video_capture.release()
        video_capture = None

    return JSONResponse({"detail": "Video stream stopped"})

@app.get("/latest_plate")
def latest_plate():
    global latest_detected_plate
    return {"license_plate": latest_detected_plate or "No plate detected yet"}

@app.get("/video_page", response_class=HTMLResponse)
def video_page():
    """HTML Page with live video and auto-close detection"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live License Plate Detection</title>
    </head>
    <body>
        <h1>Live License Plate Recognition</h1>
        <img src="/video_feed" width="640" height="480" />
        <div id="result"><h3>License Plate: No plate detected yet</h3></div>
        
        <script>
            async function fetchLicensePlate() {
                const response = await fetch('/latest_plate');
                const data = await response.json();
                document.getElementById('result').innerHTML = `<h3>License Plate: ${data.license_plate}</h3>`;
            }
            setInterval(fetchLicensePlate, 2000);

            // Detect page unload and stop video streaming
            window.addEventListener("beforeunload", async function () {
                navigator.sendBeacon("/stop_video"); // Ensures request is sent before closing
            });
        </script>
    </body>
    </html>
    """)

@app.get("/drivers", response_class=HTMLResponse)
async def drivers_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Driver Management</title>
      <style>
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
        table {
          width: 100%;
          margin-top: 20px;
          border-collapse: collapse;
          background: white;
          color: black;
          border-radius: 5px;
        }
        th, td {
          padding: 10px;
          border: 1px solid black;
          text-align: left;
        }
        input, button {
          display: block;
          width: 100%;
          margin: 10px 0;
          padding: 10px;
          border-radius: 5px;
          border: none;
        }
        button {
          background-color: #007BFF;
          color: #fff;
          cursor: pointer;
        }
        button:hover {
          background-color: #0056b3;
        }
        .modal {
          display: none;
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: white;
          padding: 20px;
          border-radius: 5px;
          box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
          z-index: 1000;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Driver Management</h1>
        <form id="driverForm">
          <input type="text" id="license_plate" placeholder="License Plate" required>
          <input type="text" id="name" placeholder="Driver Name" required>
          <button type="submit">Add Driver</button>
        </form>
        <h2>Drivers List</h2>
        <table>
          <thead>
            <tr>
              <th>License Plate</th>
              <th>Name</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="driverList"></tbody>
        </table>
      </div>
      <div id="updateModal" class="modal">
        <h2>Update Driver</h2>
        <input type="hidden" id="update_license_plate">
        <input type="text" id="update_name" placeholder="New Name" required>
        <button onclick="confirmUpdate()">Update</button>
        <button onclick="closeModal()">Cancel</button>
      </div>
      <script>
        async function fetchDrivers() {
          const response = await fetch('/get_drivers');
          const data = await response.json();
          const list = document.getElementById('driverList');
          list.innerHTML = '';
          data.forEach(driver => {
            list.innerHTML += `<tr>
              <td>${driver.license_plate}</td>
              <td>${driver.name}</td>
              <td>
                <button onclick="deleteDriver('${driver.license_plate}')">Delete</button>
                <button onclick="openModal('${driver.license_plate}', '${driver.name}')">Update</button>
              </td>
            </tr>`;
          });
        }
        function openModal(license_plate, name) {
          document.getElementById('update_license_plate').value = license_plate;
          document.getElementById('update_name').value = name;
          document.getElementById('updateModal').style.display = 'block';
        }
        function closeModal() {
          document.getElementById('updateModal').style.display = 'none';
        }
        async function confirmUpdate() {
          const license_plate = document.getElementById('update_license_plate').value;
          const name = document.getElementById('update_name').value;
          await fetch('/update_driver', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ license_plate, name })
          });
          closeModal();
          fetchDrivers();
        }
        async function deleteDriver(license_plate) {
          await fetch(`/delete_driver?license_plate=${license_plate}`, { method: 'DELETE' });
          fetchDrivers();
        }
        document.getElementById('driverForm').addEventListener('submit', async (e) => {
          e.preventDefault();
          const license_plate = document.getElementById('license_plate').value;
          const name = document.getElementById('name').value;
          await fetch('/register_driver', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ license_plate, name })
          });
          fetchDrivers();
        });
        fetchDrivers();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/get_drivers")
def get_drivers():
    session = SessionLocal()
    drivers = session.query(Driver).all()
    session.close()
    return [{"id": d.id, "license_plate": d.license_plate, "name": d.name} for d in drivers]

@app.post("/register_driver")
def register_driver(driver_data: dict):
    session = SessionLocal()
    new_driver = Driver(license_plate=driver_data["license_plate"], name=driver_data["name"])
    session.add(new_driver)
    session.commit()
    session.close()
    return {"detail": "Driver added successfully"}

@app.put("/update_driver")
def update_driver(driver_data: dict):
    session = SessionLocal()
    driver = session.query(Driver).filter(Driver.license_plate == driver_data["license_plate"]).first()
    if driver:
        driver.name = driver_data["name"]
        session.commit()
    session.close()
    return {"detail": "Driver updated successfully"}

@app.delete("/delete_driver")
def delete_driver(license_plate: str):
    session = SessionLocal()
    driver = session.query(Driver).filter(Driver.license_plate == license_plate).first()
    if driver:
        session.delete(driver)
        session.commit()
    session.close()
    return {"detail": "Driver deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)