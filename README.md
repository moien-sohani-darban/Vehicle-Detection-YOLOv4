<div align="center">

# 🚗 Vehicle Detection Using YOLOv4

### Vehicle detection in video footage using Python, OpenCV DNN, and YOLOv4

A lightweight computer vision project that uses a pretrained YOLOv4 model to detect objects in video frames and visualize detections using bounding boxes.

<br>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-DNN-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv4](https://img.shields.io/badge/Model-YOLOv4-black?style=for-the-badge)](https://github.com/AlexeyAB/darknet)
[![Git LFS](https://img.shields.io/badge/Git-LFS-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-lfs.com/)

<br>

[Overview](#-overview) •
[Features](#-features) •
[How-It-Works](#-how-it-works) •
[Installation](#-installation) •
[Usage](#-usage) •
[Project-Structure](#-project-structure)

</div>

---

## 📌 Overview

This project demonstrates object detection in video footage using **YOLOv4** and OpenCV's Deep Neural Network module.

The application loads a pretrained YOLOv4 model, processes the input video frame by frame, detects objects, and visualizes the detected regions using bounding boxes.

The project is implemented in Python and uses OpenCV DNN for inference.

---

## ✨ Features

- 🎥 Video-based object detection
- 🧠 Pretrained YOLOv4 model
- 📦 Bounding-box visualization
- ⚡ OpenCV DNN inference
- 🚀 CUDA GPU backend support
- 🎯 Configurable confidence threshold
- 🧹 Non-Maximum Suppression (NMS)
- 🖥️ Real-time frame visualization
- 📦 YOLO model weights managed using Git LFS

---

## 🧠 How It Works

The application follows a simple computer vision pipeline:

```text
┌────────────────────┐
│    Input Video     │
│     Cars.mp4       │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Read Video Frame │
│      OpenCV        │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│       YOLOv4       │
│   Object Detection │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Confidence + NMS   │
│     Filtering      │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Bounding Boxes   │
│   Visualization    │
└────────────────────┘
```

### 1. Model Loading

The YOLOv4 network is loaded using:

```python
cv2.dnn.readNet(weights_path, cfg_path)
```

The project uses:

```text
dnn_model/yolov4.weights
dnn_model/yolov4.cfg
```

---

### 2. CUDA Backend

The OpenCV DNN module is configured to use CUDA:

```python
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

This allows inference to run on a compatible NVIDIA GPU.

---

### 3. Detection

Each frame is passed to OpenCV's DNN detection model:

```python
class_ids, scores, boxes = od.detect(frame)
```

The detection model uses:

```text
Confidence Threshold: 0.5
NMS Threshold:        0.4
Input Resolution:     608 × 608
```

---

### 4. Visualization

Detected objects are visualized by drawing rectangular bounding boxes directly onto the video frame.

The processed frame is then displayed using OpenCV.

---

## 🏗️ Architecture

```text
                ┌─────────────────────┐
                │      Cars.mp4       │
                │    Input Video      │
                └──────────┬──────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Object_Tracking.py    │
              │                        │
              │  Video Capture         │
              │  Frame Processing      │
              │  Visualization         │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Object_Detection.py    │
              │                        │
              │ YOLOv4 Model Loading   │
              │ OpenCV DNN             │
              │ CUDA Configuration     │
              │ Object Detection       │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Detection Results      │
              │ Bounding Boxes         │
              └────────────────────────┘
```

---

## 📁 Project Structure

```text
Vehicle-Detection-YOLOv4/
│
├── Object_Detection.py
│   └── YOLOv4 model loading and object detection
│
├── Object_Tracking.py
│   └── Video processing and bounding-box visualization
│
├── Cars.mp4
│   └── Sample input video
│
├── dnn_model/
│   ├── yolov4.cfg
│   └── yolov4.weights
│
├── .gitattributes
│   └── Git LFS configuration
│
└── README.md
    └── Project documentation
```

---

## 🛠️ Technologies

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core programming language |
| **OpenCV** | Video processing and visualization |
| **OpenCV DNN** | Neural network inference |
| **YOLOv4** | Object detection model |
| **NumPy** | Numerical operations |
| **CUDA** | GPU-accelerated inference |
| **Git LFS** | Large model-weight storage |

---

## 🚀 Getting Started

### Prerequisites

Before running the project, make sure you have:

- Python 3
- OpenCV
- NumPy
- Git LFS
- A CUDA-compatible OpenCV environment if GPU acceleration is used

---

## 📥 Clone the Repository

```bash
git clone https://github.com/MoienSohaniDarban/Vehicle-Detection-YOLOv4.git
```

Navigate into the project:

```bash
cd Vehicle-Detection-YOLOv4
```

---

## 📦 Git LFS

The YOLOv4 model weights are stored using **Git Large File Storage** because the weights file is larger than GitHub's standard file-size limit.

Install Git LFS:

```bash
git lfs install
```

Download the LFS-managed files:

```bash
git lfs pull
```

---

## 📦 Install Dependencies

Install the required Python packages:

```bash
pip install opencv-python numpy
```

---

## ▶️ Run the Project

Run:

```bash
python Object_Tracking.py
```

The application will open the sample video and display detected objects using bounding boxes.

Press:

```text
Q
```

to close the video window.

---

## ⚙️ Detection Configuration

The detection configuration is defined in `Object_Detection.py`.

```python
self.nmsThreshold = 0.4
self.confThreshold = 0.5
self.image_size = 608
```

### Confidence Threshold

```text
0.5
```

Controls the minimum confidence required for a detection to be accepted.

### NMS Threshold

```text
0.4
```

Non-Maximum Suppression helps eliminate overlapping duplicate bounding boxes.

### Input Size

```text
608 × 608
```

The YOLOv4 network processes input frames at this resolution.

---

## 🧩 Model Files

The project uses the standard YOLOv4 configuration and weights:

```text
dnn_model/
├── yolov4.cfg
└── yolov4.weights
```

Because `yolov4.weights` is a large binary file, it is versioned using Git LFS.

---

## 🎯 Project Purpose

This project was developed to explore the fundamentals of computer vision and object detection using a pretrained deep-learning model.

The project demonstrates concepts including:

- Video frame processing
- Deep neural network inference
- YOLO-based detection
- Bounding-box visualization
- Confidence filtering
- Non-Maximum Suppression
- GPU-accelerated inference with CUDA

---

## 📊 Processing Flow

```text
Video
  │
  ▼
Frame Extraction
  │
  ▼
YOLOv4 Detection
  │
  ▼
Confidence Filtering
  │
  ▼
Non-Maximum Suppression
  │
  ▼
Bounding Box Rendering
  │
  ▼
Output Frame
```

---

## ⚠️ Notes

The project currently uses a fixed sample video:

```text
Cars.mp4
```

and expects the YOLOv4 model files to be located inside:

```text
dnn_model/
```

The OpenCV DNN backend is configured for CUDA acceleration, so the runtime environment should support OpenCV CUDA DNN execution.

---

## 🗺️ Possible Future Improvements

Potential future enhancements could include:

- [ ] Vehicle-specific class filtering
- [ ] Persistent vehicle IDs
- [ ] Multi-object tracking
- [ ] Vehicle counting
- [ ] Speed estimation
- [ ] Direction detection
- [ ] Tracking trajectories
- [ ] Output-video export
- [ ] Webcam input
- [ ] Real-time camera streams
- [ ] DeepSORT / ByteTrack integration
- [ ] Configurable command-line input
- [ ] CPU fallback support

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/your-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push the branch

```bash
git push origin feature/your-feature
```

5. Open a Pull Request

---

## 👤 Author

<div align="center">

### Moien Sohani

[![GitHub](https://img.shields.io/badge/GitHub-MoienSohani-181717?style=for-the-badge&logo=github)](https://github.com/moien-sohani-darban)

</div>

---

<div align="center">

### ⭐ If you find this project useful, consider giving it a star.

**Built with Python, OpenCV, and YOLOv4**

</div>
