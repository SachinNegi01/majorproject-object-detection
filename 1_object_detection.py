# import streamlit as st
# import cv2
# from ultralytics import YOLO
# import numpy as np

# st.title("📷 YOLO Object Detection")

# @st.cache_resource
# def load_model():
#     return YOLO("my_yolo_2.pt")

# model = load_model()

# run = st.checkbox("Start Camera")

# FRAME_WINDOW = st.image([])

# cap = cv2.VideoCapture()

# while run:
#     ret, frame = cap.read()
    
#     if not ret:
#         st.error("Failed to access camera")
#         break

#     results = model(frame)

#     annotated_frame = results[0].plot()

#     # Convert BGR → RGB (important)
#     annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

#     FRAME_WINDOW.image(annotated_frame)

# cap.release()

import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("📷 Real-Time YOLO Object Detection")

# -------------------------------
# Load YOLO Model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("my_yolo_2.pt")

model = load_model()

# -------------------------------
# Video Processing Class
# -------------------------------
class VideoProcessor(VideoTransformerBase):

    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        results = model(img)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        return annotated_frame

# -------------------------------
# Start Webcam Stream
# -------------------------------
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

st.success("Webcam is running...")