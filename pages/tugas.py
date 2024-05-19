import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image

def calculate_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    alpha = 1 + contrast / 127
    beta = brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def find_contours(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray_image, 127, 255, 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    st.set_page_config(layout="wide")
    st.title("Image Manipulation Web App")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)


        with col2:
            st.sidebar.subheader("Image Manipulations")
            selected_option = st.sidebar.selectbox("Select an option", ["RGB to HSV", "Histogram", "Brightness and Contrast", "Contour", "Grayscale", "Blur", "Edge Detection", "Thresholding", "Rotate", "Resize", "Flip", "Crop"], index=None)

            manipulated_image = None

            if selected_option == "RGB to HSV":
                hsv_image = rgb_to_hsv(image)
                manipulated_image = hsv_image

            elif selected_option == "Histogram":
                histogram = calculate_histogram(image)
                st.bar_chart(histogram)

            elif selected_option == "Brightness and Contrast":
                brightness = st.slider("Brightness", -100, 100, 0)
                contrast = st.slider("Contrast", -100, 100, 0)
                adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
                manipulated_image = adjusted_image

            elif selected_option == "Contour":
                contours = find_contours(image)
                image_with_contours = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
                manipulated_image = image_with_contours

            elif selected_option == "Grayscale":
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                manipulated_image = grayscale_image

            elif selected_option == "Blur":
                blur_image = cv2.GaussianBlur(image, (5, 5), 0)
                manipulated_image = blur_image

            elif selected_option == "Edge Detection":
                edges = cv2.Canny(image, 100, 200)
                manipulated_image = edges

            elif selected_option == "Thresholding":
                _, threshold_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY)
                manipulated_image = threshold_image

            elif selected_option == "Rotate":
                angle = st.slider("Angle", -180, 360, 0)
                rows, cols = image.shape[:2]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                rotated_image = cv2.warpAffine(image, M, (cols, rows))
                manipulated_image = rotated_image

            elif selected_option == "Resize":
                width = st.number_input("Width", value=image.shape[1])
                height = st.number_input("Height", value=image.shape[0])
                resized_image = cv2.resize(image, (int(width), int(height)))
                manipulated_image = resized_image

            elif selected_option == "Flip":
                flip_direction = st.radio("Flip Direction", ["Vertical", "Horizontal"])
                if flip_direction == "Vertical":
                    flipped_image = cv2.flip(image, 0)
                else:
                    flipped_image = cv2.flip(image, 1)
                manipulated_image = flipped_image

            elif selected_option == "Crop":
                x = st.number_input("X coordinate", value=0)
                y = st.number_input("Y coordinate", value=0)
                width = st.number_input("Width", value=image.shape[1])
                height = st.number_input("Height", value=image.shape[0])
                cropped_image = image[y:y+height, x:x+width]
                manipulated_image = cropped_image

            if manipulated_image is not None:
                st.image(manipulated_image, caption="Manipulated Image", use_column_width=True)

                # Convert manipulated image to appropriate format
                if len(manipulated_image.shape) == 2:  # Grayscale or single channel
                    manipulated_image = cv2.cvtColor(manipulated_image, cv2.COLOR_GRAY2RGB)
                else:
                    manipulated_image = cv2.cvtColor(manipulated_image, cv2.COLOR_BGR2RGB)
                
                pil_image = Image.fromarray(manipulated_image)
                buf = BytesIO()
                pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="manipulated_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
