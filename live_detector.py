import cv2
import easyocr
import math
import os
import csv
import time
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
CONF_THRESHOLD = 0.50 
OCR_THRESHOLD = 0.50    
VIDEO_FILE = "test2.mp4"
MODEL_FILE = "true_best.pt"
SAVE_COOLDOWN = 3.0 # Wait 3 seconds before saving the next image to prevent spam

# --- 1. SETUP LOCAL DATABASE (CSV) ---
if not os.path.exists('evidence'):
    os.makedirs('evidence')

csv_file = 'traffic_log.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Violation_Type', 'Plate_Number', 'Image_Path'])

# --- 2. SETUP MODELS ---
if os.path.exists(MODEL_FILE):
    model = YOLO(MODEL_FILE)
    print(f"✅ LOADING BRAIN: {MODEL_FILE}")
else:
    print("❌ ERROR: Brain NOT FOUND.")
    exit()

print("✅ LOADING OCR...")
try:
    reader = easyocr.Reader(['en'], gpu=True) 
except:
    reader = None

cap = cv2.VideoCapture(VIDEO_FILE)
classNames = model.names
last_save_time = 0

def save_evidence(violation_type, plate_text, frame):
    """Saves image locally and writes to CSV"""
    global last_save_time
    if time.time() - last_save_time < SAVE_COOLDOWN:
        return # Skip if we just saved one

    # Generate names
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_name = f"evidence/{timestamp}_{violation_type}.jpg"
    
    # Save Image
    cv2.imwrite(img_name, frame)
    
    # Save to CSV Database
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, violation_type, plate_text, img_name])
    
    print(f"🚨 LOGGED: {violation_type} | Plate: {plate_text} -> Saved to {csv_file}")
    last_save_time = time.time()

# --- 3. MAIN LOOP ---
while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    img = cv2.resize(img, (640, 480))
    results = model(img, stream=True, conf=CONF_THRESHOLD)

    violation_found = False
    current_plate = "Unknown"

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            color = (255, 0, 0) 
            label = currentClass

            # 1. VIOLATION (Matches your specific classes)
            if currentClass in ['no_helmet', 'Non-helmet', 'No-Helmet', 'No Helmet']:
                color = (0, 0, 255) # RED
                label = "VIOLATION"
                violation_found = True

            # 2. SAFE
            elif currentClass in ['helmet', 'Helmet']:
                color = (0, 255, 0) # GREEN
                label = "SAFE"

            # 3. PLATE
            elif currentClass in ['License_Plate', 'licence_plate', 'license plate', 'Number Plate']:
                color = (0, 255, 255) # YELLOW
                label = "Plate"
                if reader and conf > OCR_THRESHOLD:
                    try:
                        plate_crop = img[y1:y2, x1:x2]
                        text = reader.readtext(plate_crop, detail=0)
                        if len(text) > 0:
                            current_plate = text[0].upper().replace(" ", "")
                            cv2.putText(img, f"[{current_plate}]", (x1, y1 - 25), 
                                        cv2.FONT_HERSHEY_BOLD, 0.8, (0, 255, 255), 2)
                    except:
                        pass

            # Draw
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'{label} {conf}', (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # TRIGGER SAVE
    if violation_found:
        save_evidence("No_Helmet", current_plate, img)

    cv2.imshow("Smart City ANPR System", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()