import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Urban Sematic Segmentation",
    page_icon="cityscape",
    layout="wide",
    initial_sidebar_state="auto",
)

# defines an h1 header
st.title("Urban Semantic Segmentation :cityscape: :car: :motorway: 🌳 🧍")

# displays a file uploader widget


image = st.sidebar.file_uploader(
    "Chose an image file to upload", type=["png", "jpg", "jpeg", "WebP", "bmp", "tiff"]
)
res = None


if image is not None:
    st.header("input image")
    st.image(image, use_column_width=True)

else:
    st.header("Your image will be displayed here once you upload it")

if st.sidebar.button(
    "Predict",
    help="Click on this button to run segmentation model on the uploaded image",
):
    if image is not None:
        files = {"file": image.getvalue()}
        res = requests.post(f"http://backend:8080/predict", files=files)
        paths = res.json()
    else:
        st.sidebar.warning("you need to upload an image first!")


if res is not None:
    overlay = Image.open(paths.get("overlay_name"))
    st.header("Image with segmentaiton mask")
    st.image(overlay, use_column_width=True)
    mask = Image.open(paths.get("mask_name"))
    st.header("Segmentation mask")
    st.image(mask, use_column_width=True)
    st.sidebar.header("Mask colors legends")
    legend = Image.open(paths.get("legend_name"))
    st.sidebar.image(legend, width=200)
else:
    st.header("Segmented image will be displayed here")    
