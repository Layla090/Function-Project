import time
import random
import threading

import av
import cv2
import mediapipe as mp
print(mp.__file__)
print(dir(mp))
# NumPy is a library for working with arrays and matrices, which are essential for image processing tasks.
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="One with the Function", layout="centered")

mp_pose = mp.solutions.pose

# set up the graphs
st.title("Graphs!")

# defines function to create graph images, input: graph_type (string) output: graph image. so later you can call this function with different graph types to generate different graphs. For example, you could call make_graph_image("horizontal") to create a horizontal graph, or make_graph_image("increasing") to create an increasing graph.
def make_graph_image(graph_type: str):
    # Create a new figure and axis for plotting (4, 4) is the size of the figure in inches. This creates a square figure that is 4 inches wide and 4 inches tall.
    fig, ax = plt.subplots(figsize=(4, 4))

    # Generate 200 evenly spaced values between -5 and 5 for the x-axis.
    x = np.linspace(-5, 5, 200)

    # now we will create different types of graphs based on the input graph_type. The function checks the value of graph_type and generates the corresponding graph using Matplotlib. Each graph type corresponds to a specific mathematical function or pattern that is plotted on the axes. yay!!
    if graph_type == "increasing":
        y = x
        title = ("Recreate: y = x (increasing line)")
    elif graph_type == "decreasing":
        y = -x
        title = ("Recreate: y = -x (decreasing line)")

#past here im pretty sure im gonna delete so here is where I draw the line lalala

img = st.camera_input("Recreate the Graph with Your Arms!")

st.write(type(img))

if img is not None:
    st.image(img)
    from PIL import Image

#Took file from Streamlit and turned it into an image object
    image = Image.open(img)
    image = image.convert("RGB")
# Turned image NumPy array (matrix of numbers)
    image_np = np.array(image)
    image_np = image_np[:, :, :3]

    st.write("Shape:", image_np.shape)
# [rows, columns]
# [height, width, color channels]
    st.write("Top-left pixel value:", image_np[0, 0])

# mp_pose.Pose(...) is the AI model that detects human poses in images. The "with" statement ensures that the model is properly initialized and released after use.
# The static_image_mode=True argument indicates that the model should treat the input as a static image, which is suitable for processing single images rather than video streams.
    with mp_pose.Pose(static_image_mode=True) as pose:
        image_np.flags.writeable = False
        results = pose.process(image_np)

    if results.pose_landmarks:
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        st.write("Left wrist x:", wrist.x)
        st.write("Left wrist y:", wrist.y)
    else:
        st.write("No Pose Detected.")