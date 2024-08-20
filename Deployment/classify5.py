import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
import numpy as np
import cv2
import PySpin
from pymodbus.client.sync import ModbusTcpClient
from ultralytics import YOLO
from PIL import Image

# Modbus configuration
MODBUS_HOST = '192.168.0.252'  # Replace with your PLC IP address
MODBUS_PORT = 502  # Modbus port (usually 502)
COIL_ADDRESSES = [9, 10, 11, 12, 13]


# Function to initialize the camera
def init_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        raise RuntimeError('No cameras detected!')

    cam = cam_list[0]
    cam.Init()

    nodemap = cam.GetNodeMap()
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

    cam.BeginAcquisition()
    return cam, system, cam_list


# Function to process the image
def process_image(img, model):
    img_data = img.GetData()
    width = img.GetWidth()
    height = img.GetHeight()
    img_size = len(img_data)

    pixel_format = img.GetPixelFormat()
    bytes_per_pixel = 1

    if pixel_format == PySpin.PixelFormat_BayerRG8:
        bytes_per_pixel = 1
    elif pixel_format == PySpin.PixelFormat_RGB8:
        bytes_per_pixel = 3
    elif pixel_format == PySpin.PixelFormat_MONO8:
        bytes_per_pixel = 1
    elif pixel_format == PySpin.PixelFormat_MONO16:
        bytes_per_pixel = 2
    else:
        raise ValueError(f'Unsupported pixel format: {pixel_format}')

    expected_size = width * height * bytes_per_pixel
    if img_size != expected_size:
        raise ValueError(f'Size mismatch: {img_size} != {expected_size}')

    img_array = np.array(img_data, dtype=np.uint8).reshape(height, width, bytes_per_pixel)

    if pixel_format == PySpin.PixelFormat_BayerRG8:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_BAYER_RG2BGR)
    elif pixel_format == PySpin.PixelFormat_RGB8:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif pixel_format == PySpin.PixelFormat_MONO8:
        img_cv = img_array
    else:
        raise ValueError(f'Unsupported pixel format for conversion: {pixel_format}')

    results = model(img_cv)
    return img_cv, results


# Function to update Modbus coils based on predictions
def update_modbus_coils(client, predictions):
    max_prob_index = np.argmax(predictions)
    prob = predictions[max_prob_index]

    for i, address in enumerate(COIL_ADDRESSES):
        value = 1 if (i == max_prob_index and prob > 0.90) else 0
        client.write_coil(address, value)


def main():
    # Customizing the app's layout and style
    st.set_page_config(page_title="YOLOv8 Camera Stream", page_icon=":camera:", layout="wide")

    # Add custom CSS to style the Streamlit app
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .header {
            font-size: 2em;
            color: #007BFF;
            text-align: center;
            margin-bottom: 20px;
        }
        .content {
            display: flex;
            justify-content: space-around;
            align-items: flex-start;
            gap: 20px;
        }
        .image-container {
            flex: 1;
        }
        .results-container {
            flex: 1;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header">YOLOv8 Camera Stream</div>', unsafe_allow_html=True)

    # Initialize camera and model
    cam, system, cam_list = init_camera()
    model = YOLO('model_3.pt')

    st.title("YOLOv8 Camera Stream")

    MODBUS_HOST = '192.168.0.252'
    MODBUS_PORT = 502


    # Initialize Modbus client
    client = ModbusTcpClient(MODBUS_HOST, port=MODBUS_PORT)
    client.connect()

    try:
        # Create layout containers
        col1, col2 = st.columns([2, 1])

        with col1:
            image_placeholder = st.empty()

        with col2:
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            class_placeholders = [st.empty() for _ in range(5)]
            st.markdown('</div>', unsafe_allow_html=True)

        while True:
            try:

                img = cam.GetNextImage()
                if img.IsIncomplete():
                    st.write(f'Image incomplete with status {img.GetImageStatus()}')
                    img.Release()
                    continue

                img_cv, results = process_image(img, model)

                image_placeholder.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), channels="RGB")

                # Clear previous class probability placeholders
                for placeholder in class_placeholders:
                    placeholder.empty()

                # Process and display results
                if results is not None:
                    result = results[0]
                    class_names = result.names
                    probabilities = result.probs.data.numpy()

                    # Display probabilities for the fixed number of classes
                    for idx in range(min(5, len(class_names))):  # Ensure not to exceed available class names
                        class_name = class_names[idx]
                        prob = probabilities[idx]
                        class_placeholders[idx].write(f"Class: {class_name}, Probability: {prob:.4f}")

                    # Update Modbus coils
                    update_modbus_coils(client, probabilities)

                img.Release()

            except PySpin.SpinnakerException as ex:
                st.write(f'Error: {ex}')
                break

    finally:
        try:
            cam.EndAcquisition()
            cam.DeInit()
        except PySpin.SpinnakerException as ex:
            st.write(f'Error ending acquisition: {ex}')

        del cam
        cam_list.Clear()
        system.ReleaseInstance()

        # Close Modbus client
        client.close()


if __name__ == '__main__':
    main()
