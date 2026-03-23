import streamlit as st
import mediapipe as mp
# NumPy is a library for working with arrays and matrices, which are essential for image processing tasks.
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="One with the Function", layout="centered")

mp_pose = mp.solutions.pose

if "page" not in st.session_state:
    st.session_state.page = "home"

def change_page(new_page):
    st.session_state.page = new_page

def home():
    st.title("One with the Func")
    st.write("by Sara Koka")

    if st.button("Level Mode"):
        change_page("graph")

# defines function to create graph images, input: graph_type (string) output: graph image. so later you can call this function with different graph types to generate different graphs. For example, you could call make_graph_image("horizontal") to create a horizontal graph, or make_graph_image("increasing") to create an increasing graph.
def make_graph_image(graph_type: str):
    # set up the graphs
    st.title("Memorize the Shape of the Graph")

    # Create a new figure and axis for plotting (4, 4) is the size of the figure in inches. This creates a square figure that is 4 inches wide and 4 inches tall.
    fig, ax = plt.subplots()
    ax.plot(x, y, label="points")

    # Generate 200 evenly spaced values between -5 and 5 for the x-axis.
    x = np.linspace(-5, 5, 200)
    y == np.linspace(-5, 5, 200)

    # now we will create different types of graphs based on the input graph_type. The function checks the value of graph_type and generates the corresponding graph using Matplotlib. Each graph type corresponds to a specific mathematical function or pattern that is plotted on the axes. yay!!
    if graph_type == "increasing linearly":
        y = x
        ax.set_title("Recreate: y = x (increasing linear graph)")
    elif graph_type == "decreasing linearly":
        y = -x
        ax.set_title("Recreate: y = -x (decreasing linear graph)")
    elif graph_type == "positive absolute value":
        y = abs(x)
        ax.set_title("Recreate: y = |x| (positive absolute value graph)")
    elif graph_type == "positive quadratic":
        y = (x**2)
        ax.set_title("Recreate: y = x^2 (positive quadratic)")
    plt.show()

if st.session_state.page == "home":
    home()
elif st.session_state.page == "graph":
    make_graph_image("increasing linearly")