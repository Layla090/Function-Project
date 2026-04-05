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

    x = np.linspace(0, 2, 100)  # Sample data.

    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(x, x, label='linear')  # Plot some data on the (implicit) Axes.
    plt.plot(x, x**2, label='quadratic')  # etc.
    plt.plot(x, x**3, label='cubic')
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()

if st.session_state.page == "home":
    home()
elif st.session_state.page == "graph":x