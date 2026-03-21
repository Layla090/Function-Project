import mediapipe as mp
import streamlit as st

mp_pose = mp.solutions.pose

st.title("Pose 1")

img = st.camera_input("Recreate the Graph with Your Arms!")

st.write(type(img))

if img:
    st.image(img)
    from PIL import Image
# NumPy is a library for working with arrays and matrices, which are essential for image processing tasks.
    import numpy as np
#Took file from Streamlit and turned it into an image object
    image = Image.open(img)
# Turned image NumPy array (matrix of numbers)
    image_np = np.array(image)

    st.write("Shape:", image_np.shape)
# [rows, columns]
# [height, width, color channels]
    st.write("Top-left pixel value:", image_np[0, 0])

# mp_pose.Pose(...) is the AI model that detects human poses in images. The "with" statement ensures that the model is properly initialized and released after use.
# The static_image_mode=True argument indicates that the model should treat the input as a static image, which is suitable for processing single images rather than video streams.
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)

    if results.pose_landmarks:
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        st.write("Left wrist x:", wrist.x)
        st.write("Left wrist y:", wrist.y)
    else:
        st.write("No Pose Detected.")