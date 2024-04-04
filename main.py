from ultralytics import YOLO
import cv2
from utils import write_csv, read_license_plate, get_car
import datetime
import numpy as np
from sort.sort import *
import pymongo
from dotenv import load_dotenv
# Connect to MongoDB

load_dotenv()

try:
    client = pymongo.MongoClient('DB_URL')
    db = client['PlateRecognitionApp']  
    collection = db['license_plate']
except  pymongo.errors.ConnectionFailure as e:
    raise Exception(f"Failed to connect to MongoDB: {e}")

# Function to add or update data in MongoDB based on score
def add_or_update_data(car):
    car_id = car['car_id']
    existing_data = collection.find_one({'car_id': car_id})
    
    if existing_data:
        current_time = datetime.datetime.now()
        existing_time = existing_data['time']
        time_difference = (current_time - existing_time).total_seconds() / 60  # Calculate time difference in minutes
        
        if car['license_plate_score'] > existing_data['license_plate_score'] or time_difference >= 5:
            update_data = {
                'license_plate_number': car['license_plate_number'],
                'license_plate_score': car['license_plate_score'],
                'time': current_time
            }
            collection.update_one({'car_id': car_id}, {'$set': update_data})
    else:
        data = {
            'car_id': car_id,
            'time': car['time'],
            'license_plate_number': car['license_plate_number'],
            'license_plate_score': car['license_plate_score'],
        }
        collection.insert_one(data)



vehicle_tracker = Sort()

# Load model
license_plate_detector = YOLO('./model/best.pt')
coco_model = YOLO('yolov8n.pt')

# Load video
cap = cv2.VideoCapture('./videos/sample2.mp4')

vehicles = [2, 3, 5, 7]

# Read frames
ret = True
frame_nmr = -1

while ret:
    ret, frame = cap.read()
    frame_nmr += 1

    if ret:
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_all = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            if int(class_id) in vehicles:
                detections_all.append([x1, y1, x2, y2, score])

        # Track vehicles
        if detections_all:
            track_ids = vehicle_tracker.update(np.asarray(detections_all))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            x1_car, y1_car, x2_car, y2_car, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_crop_denoised = cv2.fastNlMeansDenoising(license_plate_crop_gray, None, 10, 7, 21)
                license_plate_crop_resized = cv2.resize(license_plate_crop_denoised, (0, 0), fx=5, fy=5,
                                                        interpolation=cv2.INTER_LANCZOS4)

                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_resized, 110, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    data = {
                        'car_id': car_id,
                        'time': datetime.datetime.now(),
                        'license_plate_number': license_plate_text,
                        'license_plate_score': license_plate_text_score
                    }
                    add_or_update_data(data)
