import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Fungsi untuk menangkap gambar dari kamera
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        return None

st.title("Camera Capture")

# Menangkap gambar ketika tombol diklik
if st.button("Capture Image"):
    image = capture_image()
    if image is not None:
        st.image(image, caption="Captured Image", use_column_width=True)
        
        # Convert captured image to PIL format for saving
        pil_image = Image.fromarray(image)
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Add a download button for the captured image
        st.download_button(
            label="Download Image",
            data=byte_im,
            file_name="captured_image.png",
            mime="image/png"
        )
    else:
        st.error("Failed to capture image")
