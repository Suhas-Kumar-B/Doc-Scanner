# Title: Document Scanner Streamlit App
import streamlit as st
import tempfile
import cv2
import scanner
import numpy as np
import io
from PIL import Image


# --- Utility Function to Convert OpenCV Image to Bytes for Download ---
def cv2_to_bytes(img, format="PNG"):
    # Convert OpenCV image (numpy array) to PIL Image
    if len(img.shape) == 2:  # Grayscale image
        pil_img = Image.fromarray(img, mode="L")
    else:  # Color image (BGR)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Save to bytes buffer
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Document Scanner", layout="wide")
st.title("📄 AI Document Scanner")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Debug: Display the uploaded file details
        st.write("**Debug Info**")
        st.write(f"Uploaded file name: {uploaded_file.name}")
        st.write(f"Uploaded file type: {uploaded_file.type}")
        st.write(f"Uploaded file size: {uploaded_file.size} bytes")

        # --- Save Uploaded File to Temporary Location ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        st.write(f"Temporary file path: {file_path}")

        # --- Process the Image ---
        results = scanner.scan_document(file_path)

        # --- Display Processing Stages ---
        st.subheader("📸 Processing Stages")

        # --- Section 1: Initial Processing ---
        st.markdown("### Initial Processing")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(results["original"], channels="BGR", caption="Original Image", width=300)
        with col2:
            st.image(results["gray"], channels="GRAY", caption="Grayscale Conversion", width=300)
        with col3:
            st.image(results["blurred"], channels="GRAY", caption="Blurred Image", width=300)

        # --- Section 2: Contour Detection and Perspective Correction ---
        st.markdown("### Contour Detection and Perspective Correction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(results["edged"], channels="GRAY", caption="Canny Edge Detection", width=300)
        with col2:
            st.image(results["outline"], channels="BGR", caption="Outline Detection", width=300)
        with col3:
            st.image(results["scanned"], channels="GRAY", caption="Perspective Scanned Output", width=300)
            # Add download button for scanned image
            scanned_bytes = cv2_to_bytes(results["scanned"])
            st.download_button(
                label="⬇️ Download Scanned Output",
                data=scanned_bytes,
                file_name="scanned_output.png",
                mime="image/png"
            )

        # --- Section 3: Thresholding Variants ---
        st.markdown("### Thresholding Variants")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(results["binary_thresh"], channels="GRAY", caption="Binary Threshold", width=200)
            # Add download button for binary threshold
            binary_bytes = cv2_to_bytes(results["binary_thresh"])
            st.download_button(
                label="⬇️ Download Binary",
                data=binary_bytes,
                file_name="binary_threshold.png",
                mime="image/png"
            )
        with col2:
            st.image(results["mean_thresh"], channels="GRAY", caption="Mean Adaptive Threshold", width=200)
            # Add download button for mean threshold
            mean_bytes = cv2_to_bytes(results["mean_thresh"])
            st.download_button(
                label="⬇️ Download Mean",
                data=mean_bytes,
                file_name="mean_threshold.png",
                mime="image/png"
            )
        with col3:
            st.image(results["gaussian_thresh"], channels="GRAY", caption="Gaussian Adaptive Threshold", width=200)
            # Add download button for Gaussian threshold
            gaussian_bytes = cv2_to_bytes(results["gaussian_thresh"])
            st.download_button(
                label="⬇️ Download Gaussian",
                data=gaussian_bytes,
                file_name="gaussian_threshold.png",
                mime="image/png"
            )
        with col4:
            st.image(results["otsu_thresh"], channels="GRAY", caption="Otsu's Threshold", width=200)
            # Add download button for Otsu's threshold
            otsu_bytes = cv2_to_bytes(results["otsu_thresh"])
            st.download_button(
                label="⬇️ Download Otsu's",
                data=otsu_bytes,
                file_name="otsu_threshold.png",
                mime="image/png"
            )

        # --- Success Message ---
        st.success("✅ All document scan stages displayed successfully!")

    except FileNotFoundError as e:
        st.error(f"❌ Error: {str(e)}")
    except ValueError as e:
        st.error(f"❌ Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
else:
    st.info("Please upload an image to start scanning.")