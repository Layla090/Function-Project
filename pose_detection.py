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
    if graph_type == "increasing linearly":
        y = x
        title = ("Recreate: y = x (increasing linear graph)")
    elif graph_type == "decreasing linearly":
        y = -x
        title = ("Recreate: y = -x (decreasing linear graph)")
    elif graph_type == "positive absolute value":
        y = abs(x)
        title = ("Recreate: y = |x| (positive absolute value graph)")
    elif graph_type == "positive quadratic":
        y = (x^2)
        title = ("Recreat: y = x^2 (positive quadratic)")