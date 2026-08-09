# 🛣️ AI-Based Smart Road Damage Detection

An AI-powered web application for detecting road damage from road images using a trained YOLO-based object detection model.

## 📌 Project Overview

This project detects different types of road damage from uploaded road images and provides:

* Damage Type
* Confidence Score
* Severity Level
* Maintenance Recommendation

The application provides an interactive web interface using Streamlit.

## 🔍 Supported Damage Types

The trained model detects five damage classes:

| Class | Damage Type        |
| ----- | ------------------ |
| D00   | Longitudinal Crack |
| D10   | Transverse Crack   |
| D20   | Alligator Crack    |
| D40   | Pothole            |
| D43   | Repair             |

## ⚙️ Application Workflow

```text
Road Image
    ↓
Image Upload
    ↓
Image Preprocessing
    ↓
YOLO Model Prediction
    ↓
Damage Type
    ↓
Confidence Score
    ↓
Severity Level
    ↓
Maintenance Recommendation
    ↓
Display Results
```

## 🤖 Model

The application uses a trained YOLO object detection model.

Trained model file:

```text
best.pt
```

The model was trained for five road-damage classes.

## 📊 Application Output

For each detected damage, the application displays:

* Damage Type
* Confidence Score
* Severity
* Maintenance Recommendation
* Detection bounding box

### Example

```text
Damage Type: Transverse Crack
Confidence: 52.65%
Severity: Medium
Maintenance Recommendation: Schedule Maintenance
```

## 🛠️ Technologies Used

* Python
* Ultralytics YOLO
* Streamlit
* OpenCV
* NumPy
* Pillow

## 📁 Project Structure

```text
Road-Damage-Detection/
│
├── app.py
├── test_model.py
├── best.pt
├── last.pt
├── dataset.yaml
├── requirements.txt
├── README.md
│
├── test_images/
│   └── China_Drone_000008.jpg
│
└── runs/
```

## 🚀 How to Run Locally

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Install dependencies

Open the terminal and run:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

### 4. Upload an Image

Upload a JPG, JPEG, or PNG road image through the application.

The application will display the detected damage, confidence score, severity, and maintenance recommendation.

## 📋 Severity Levels

The application provides three severity levels:

* Low
* Medium
* High

The severity is estimated using the detected damage bounding-box area and damage type.

## 🔧 Maintenance Recommendations

| Severity | Recommendation               |
| -------- | ---------------------------- |
| Low      | No Immediate Action Required |
| Medium   | Schedule Maintenance         |
| High     | Immediate Repair Required    |

## 🎯 Project Objective

The objective of this project is to develop an AI-based road damage detection application that can assist in identifying road damage and providing an estimated severity and maintenance recommendation through an easy-to-use web interface.

## 👩‍💻 Project Status

* Model Training: Completed
* Model Testing: Completed
* Streamlit Application: Completed
* Severity Assessment: Completed
* Maintenance Recommendation: Completed
* Requirements File: Completed
* README: Completed
* GitHub Deployment: Pending
* Hugging Face Deployment: Pending
* Demonstration Video: Pending
