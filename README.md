# Face Detection and Nationality Prediction

## Overview
This project leverages OpenCV and machine learning to perform real-time face detection and nationality prediction using a live camera feed. The system is designed to detect faces in the camera feed, extract features, and predict the nationality of the detected faces based on pre-trained models.

## Features
- **Real-Time Face Detection**: Utilizes OpenCV's Haar Cascade classifier to detect faces in real-time using a live camera feed.
- **Feature Extraction**: Extracts facial features to be used for classification.
- **Nationality Prediction**: Uses a machine learning model to predict the nationality of the detected faces.
- **Modular Code Structure**: Organized into separate modules for face detection and nationality prediction, making it easy to manage and extend.

## Installation

Install the necessary libraries:

```sh
install required libraries:
pip install opencv-python numpy scikit-learn

Clone this repository:
git clone https://github.com/Fahad-Al-Maashani/nationality-face-prediction-model.git
cd face-detection-nationality-prediction

Usage:
Face Detection
Run face_detection.py to detect faces using the live camera feed:
python face_detection.py

Nationality Prediction:

Run nationality_prediction.py to detect faces and predict nationality in real-time:
python nationality_prediction.py
