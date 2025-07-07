# AI Shirt Color Detection Pro 👕

![image](https://github.com/user-attachments/assets/f0fde56e-10fd-4a30-b624-68b028baf9cd)
![image](https://github.com/user-attachments/assets/8980dfd6-f957-4661-95ad-7cd340478da3)
![image](https://github.com/user-attachments/assets/9bfecb1c-6bf6-47e5-ba90-a70fc6f8ac26)
![image](https://github.com/user-attachments/assets/347ed451-532d-4fc1-aa61-6816652e0506)
![image](https://github.com/user-attachments/assets/c34629b9-1faf-45b2-8a60-88a9398aaa67)


Advanced computer vision system for detecting and analyzing shirt colors in images and videos using YOLO models and color clustering algorithms.

## Features ✨

- **Multiple AI Models**: Choose between YOLOv5 (fast), YOLOv8 (balanced), or DETR (accurate)
  ![image](https://github.com/user-attachments/assets/ad9493b6-9783-4cc5-a44e-447a8e90facd)

- **Color Detection Modes**:
  - Smart Auto-Grouping
  - Specific Color Hunt
  - Color Distribution Analysis
  - Similar Color Clustering
 ![image](https://github.com/user-attachments/assets/3b916ba8-d2a8-4806-9079-3974bfbcb2a2)

    
- **Advanced Color Analysis**:
  - K-Means Clustering
  - Histogram Analysis
  - Dominant Color + Context
  - Multi-Region Sampling
- **Image Processing**:
  - Auto-enhancement
  - Background removal (optional)
  - Shirt region focus
- **Visualizations**:
  - Color distribution charts
  - Confidence analysis
  - Color palette visualization
  - Category breakdown

![image](https://github.com/user-attachments/assets/4b038881-71d5-4a36-83aa-4381aa06afaf)

## Installation 🛠️

1. Clone the repository:
   ```bash
   git clone https://github.com/raamizhussain/Shirt-Color-Detector.git
   cd shirt-color-detection
Install dependencies:

pip install -r requirements.txt
Download YOLO weights (if not included in repo):

wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
Usage 🚀
Run the Streamlit app:

streamlit run app.py
In the web interface:

Select your preferred AI model and detection mode

Upload an image/video or use camera capture

Adjust settings as needed

View the color analysis results

## File Structure 📁
shirt-color-detection/
├── app.py                # Main application code

├── README.md             # This file

├── requirements.txt      # Python dependencies

├── yolov5s.pt            # YOLOv5 small model weights

├── Shirt_Color_Detection.ipynb  # Jupyter notebook version (optional)

└── demo.gif              # Demo animation (optional)

## Requirements 📦
Python 3.7+
Streamlit
PyTorch
OpenCV
scikit-learn
NumPy
Pillow
Matplotlib
scipy

## Contributing 🤝
Contributions are welcome! Please follow these steps:

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

## Acknowledgments 🙏
YOLOv5 and Ultralytics for the object detection models

Streamlit for the web interface framework

OpenCV for computer vision capabilities

scikit-learn for clustering algorithms
