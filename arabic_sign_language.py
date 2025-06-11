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

# Load YOLO ONNX model once
@st.cache_resource
def load_model():
    """Load and cache the ONNX model"""
    try:
        model = YOLO("model.onnx")
        st.success("✅ ONNX Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading ONNX model: {e}")
        st.info("Make sure 'model.onnx' exists in your project directory")
        return None

model = load_model()

st.title("🎯 Real-Time Arabic Language Sign Detection with YOLOv11n (ONNX)!")

# Add model info
if model:
    st.info(f"🔧 Using device: {device}")
    st.info("🚀 Model format: ONNX (Optimized for inference)")

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ==== Processor ====
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.frame_count = 0
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.model is None:
            # Return original frame if model failed to load
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, "Model not loaded!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Resize for faster detection (ONNX models work well with consistent input sizes)
        img_small = cv2.resize(img, (640, 480))
        
        try:
            # Perform inference with ONNX model
            results = self.model(img_small, conf=0.6, verbose=False)
            
            # Extract the first result
            result = results[0]
            
            # Log detections for debugging (less frequent to avoid spam)
            if result.boxes is not None and len(result.boxes) > 0 and self.frame_count % 30 == 0:
                num_detections = len(result.boxes)
                print(f"🟢 Frame {self.frame_count}: {num_detections} signs detected")
            
            # Plot annotations
            annotated_img = result.plot()
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            annotated_img = img_small
            # Add error text to frame
            cv2.putText(annotated_img, f"Error: {str(e)[:50]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Resize back to original frame size
        output = cv2.resize(annotated_img, (img.shape[1], img.shape[0]))
        
        return av.VideoFrame.from_ndarray(output, format="bgr24")

# ==== UI Controls ====
st.sidebar.header("⚙️ Settings")

# Add confidence threshold control
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.6, 
    step=0.05,
    help="Lower values detect more objects but may include false positives"
)

# Add model info in sidebar
if model:
    st.sidebar.success("🟢 Model Status: Loaded")
    st.sidebar.info("📊 Format: ONNX")
else:
    st.sidebar.error("🔴 Model Status: Failed to load")

# ==== Start Stream ====
if model:
    st.markdown("### 📹 Live Video Stream")
    st.markdown("Click **START** to begin real-time sign detection!")
    
    webrtc_streamer(
        key="sign-detection",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions:
    1. **Click START** to activate your camera
    2. **Position Arabic signs** in front of the camera
    3. **Adjust confidence threshold** in the sidebar if needed
    4. **Check console** for detection logs
    
    ### 🔧 Troubleshooting:
    - Make sure `model.onnx` is in the same directory as this script
    - Ensure good lighting for better detection
    - Try adjusting the confidence threshold
    """)
else:
    st.error("Cannot start video stream - model failed to load")
    st.markdown("""
    ### 🛠️ To fix this issue:
    1. Make sure `model.onnx` exists in your project directory
    2. Convert your `.pt` model to ONNX using: `model.export(format='onnx')`
    3. Restart the application
    """)
