# 👕 Shirt Color Detector

This is a simple Streamlit-based web app that detects shirt colors worn by people in an uploaded image using YOLOv5. Used YOLOv5 for person detection and color clustering for shirt color recognition.

### Features
- Detects multiple people in a photo
- Extracts shirt regions
- Identifies shirt colors from 100+ color names
- Clean Streamlit interface

### Usage
streamlit run app.py

## Requirements
- Python 3.8+
- torch, opencv-python, streamlit, numpy, sklearn, Pillow
