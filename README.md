# Violation & Vehicle Data Storage
An Automated Intelligent Traffic Violation Detection System built with YOLOv8 and EasyOCR.

## Overview
This edge-computing computer vision system automatically detects two-wheeler riders without helmets and extracts their license plate alphanumeric data in real-time. It logs the violation timestamps, types, and plate numbers into a local CSV database while saving photographic evidence.

## Features
* **Real-time Helmet Detection:** Custom-trained YOLOv8 model ("True_Data").
* **Automatic Number Plate Recognition (ANPR):** EasyOCR integration.
* **Automated Logging:** Generates a `traffic_log.csv` database automatically.
* **Evidence Capture:** Saves cropped `.jpg` frames of violators.
* **Anti-Spam Logic:** Prevents duplicate logs of the same violator.

## How to Run
1. Clone this repository.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Ensure `test2.mp4` and `true_best.pt` are in the root folder.
4. Run the script:
   `python live_detector.py`

## Output
The script will automatically create an `evidence/` folder containing images of violators and a `traffic_log.csv` file containing the violation database.
