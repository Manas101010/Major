import os
import streamlit as st
from predictor import check

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(APP_ROOT, './static/images/')

# Ensure the directory for saving images exists
os.makedirs(static_folder, exist_ok=True)

def upload():
    st.title('Image Uploader and Predictor')

    uploaded_files = st.file_uploader("Upload Image", type=['jpg', 'png'], accept_multiple_files=True)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            dest = os.path.join(static_folder, filename)
            with open(dest, "wb") as f:
                f.write(uploaded_file.getbuffer())
            status = check(filename)
            st.image(uploaded_file, caption=f"Uploaded Image: {filename}", use_column_width=True)
            st.write(f"Prediction for {filename}: {status}")

def main():
    upload()

if __name__ == "__main__":
    main()
