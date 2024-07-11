import streamlit as st
import torch
import shutil
from PIL import Image


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def delete_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(f"Error: {path} : {e.strerror}")

def main():
    st.title('Object Detection')
    st.write("This project uses the YOLOv5 model for real-time object detection. Users can upload an image through the interface provided, and the model will predict and display the items detected in the image.")
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        delete_folder("runs/")
        image = Image.open(uploaded_file)
        st.write("")
        st.write("Detecting...")

        result = model(image)
        result.save()

        final_image = Image.open('runs/detect/exp/image0.jpg')
        st.image(final_image, caption='Detected Image.')
        st.table(result.pandas().xyxy[0])


if __name__ == '__main__':
    main()