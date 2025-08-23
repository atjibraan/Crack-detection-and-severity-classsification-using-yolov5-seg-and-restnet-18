# 🧱 Crack Detection & Severity Classification using YOLOv5 + ResNet18

![Crack Detection](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Crack_in_pavement.jpg/640px-Crack_in_pavement.jpg)

Access the Gradio ap through https://huggingface.co/spaces/AJibraan/Crack

This project presents an end-to-end deep learning solution to automatically detect and classify cracks on concrete or pavement surfaces. We use **YOLOv5** for crack detection and **ResNet18** for severity classification. The dataset is accessed directly from **Roboflow** using their API.

---

## 📊 Overview

Manual inspection of structural surfaces like roads, bridges, and buildings is inefficient and subjective. This project automates the process:
- **YOLOv5** is used for real-time crack detection in images.
- **ResNet18** is used to classify each detected crack as **Mild**, **Moderate**, or **Severe**.

The combination of detection and classification provides a scalable system for intelligent infrastructure assessment.

---

## 🧬 Dataset

### 📦 Source: Roboflow
We used a custom annotated dataset hosted on [Roboflow](https://roboflow.com/), accessed using its API for streamlined training and testing.

Replace the placeholders below with your actual Roboflow project details.

```python
# Install Roboflow SDK
!pip install roboflow

# Access Roboflow Dataset
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("your-project-name")
dataset = project.version("1").download("yolov5")


