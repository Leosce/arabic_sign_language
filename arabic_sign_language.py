import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import torch
from ultralytics import YOLO
import os
import numpy as np

# ==== Setup ====
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model once
model = YOLO(r"C:\Users\user\arabic_signs\Models\model.pt")
model.fuse()  # Optimize model for inference

st.title("🎯 Real-Time Arabic Language Sign Detection with YOLOv11n!")

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ==== Processor ====
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Resize for faster detection
        img_small = cv2.resize(img, (640, 480))  # Try a bit larger

        # Perform inference
        results = self.model(img_small, conf=0.6)

        # Extract the first result
        result = results[0]

        # Log detections for debugging
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"🟢 Sign detected: {result}")
        

        # Try to plot directly
        try:
            annotated_img = result.plot()
        except Exception as e:
            print(f"Plotting failed: {e}")
            annotated_img = img_small

        # Resize back to original frame size
        output = cv2.resize(annotated_img, (img.shape[1], img.shape[0]))

        return av.VideoFrame.from_ndarray(output, format="bgr24")

# ==== Start Stream ====
webrtc_streamer(
    key="sign-detection",
    video_processor_factory=YOLOProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
)