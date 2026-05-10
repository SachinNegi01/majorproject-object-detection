import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.title("📷 YOLO Object Detection")

@st.cache_resource
def load_model():
    return YOLO("my_yolo_2.pt")

model = load_model()

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    
    if not ret:
        st.error("Failed to access camera")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    # Convert BGR → RGB (important)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(annotated_frame)

cap.release()