#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from collections import deque
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox

# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = DetectMultiBackend("best2.pt", device=device)

# Load ResNet-18 Crack Severity Model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Ensure consistency with training
    nn.Linear(num_ftrs, 3)  # 3 classes: Mild, Moderate, Severe
)
model.load_state_dict(torch.load("best_crack_severity_model.pth", map_location=device))
model.to(device)
model.eval()

# Severity labels and repair suggestions
severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe"}
repair_suggestions = {
    "Mild": "Regular monitoring, seal cracks with crack filler.",
    "Moderate": "Apply crack sealant, consider minor resurfacing.",
    "Severe": "Immediate structural repair needed, consult an engineer."
}

# Define transformation for severity classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Moving average buffer to smooth severity predictions
buffer_size = 10
severity_buffer = deque(maxlen=buffer_size)

# Streamlit UI
st.title("Crack Detection & Severity Classification")
input_type = st.radio("Choose input type:", ["Image", "Video"])

def predict_severity(roi):
    """Predict crack severity with smoothing."""
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(roi_tensor)
        severity_probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    severity_index = np.argmax(severity_probs)
    severity_label = severity_labels[severity_index]

    # Add to buffer and compute most frequent severity
    severity_buffer.append(severity_label)
    smoothed_severity_label = max(set(severity_buffer), key=severity_buffer.count)

    return smoothed_severity_label, severity_probs[severity_index]

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_cv = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Preprocess frame for YOLOv5
        frame_resized, ratio, (dw, dh) = letterbox(frame, 640, stride=32, auto=False)
        frame_resized = np.ascontiguousarray(frame_resized)
        img_tensor = torch.from_numpy(frame_resized).to(device).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Run YOLOv5
        detections = yolo_model(img_tensor)
        results = non_max_suppression(detections, conf_thres=0.5, iou_thres=0.3)

        detected_cracks = False
        for det in results:
            if det is not None and len(det):
                det = det.cpu().numpy()
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                for x1, y1, x2, y2, conf in det[:, :5]:
                    if conf > 0.5:
                        detected_cracks = True
                        roi = frame[int(y1):int(y2), int(x1):int(x2)]
                        if roi.size == 0:
                            continue

                        severity_label, _ = predict_severity(roi)

                        color = (0, 255, 0) if severity_label == "Mild" else (0, 255, 255) if severity_label == "Moderate" else (0, 0, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, severity_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        st.image(frame, channels="BGR")

        if detected_cracks:
            st.subheader("Repair Suggestion:")
            st.write(repair_suggestions[severity_label])

elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        st_frame = st.empty()
        detected_cracks = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized, ratio, (dw, dh) = letterbox(frame, 640, stride=32, auto=False)
            frame_resized = np.ascontiguousarray(frame_resized)
            img_tensor = torch.from_numpy(frame_resized).to(device).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            detections = yolo_model(img_tensor)
            results = non_max_suppression(detections, conf_thres=0.5, iou_thres=0.3)

            for det in results:
                if det is not None and len(det):
                    det = det.cpu().numpy()
                    det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                    for x1, y1, x2, y2, conf in det[:, :5]:
                        if conf > 0.5:
                            detected_cracks = True
                            roi = frame[int(y1):int(y2), int(x1):int(x2)]
                            if roi.size == 0:
                                continue

                            severity_label, _ = predict_severity(roi)

                            color = (0, 255, 0) if severity_label == "Mild" else (0, 255, 255) if severity_label == "Moderate" else (0, 0, 255)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, severity_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)
            st_frame.image(frame, channels="BGR")

        cap.release()
        out.release()
        st.video(output_path)

        if detected_cracks:
            dominant_severity = max(set(severity_buffer), key=severity_buffer.count)
            st.subheader("Final Repair Suggestion:")
            st.write(repair_suggestions[dominant_severity])



