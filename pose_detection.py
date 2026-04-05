import cv2
import streamlit as st
import mediapipe as mp
# NumPy is a library for working with arrays and matrices, which are essential for image processing tasks.
import numpy as np
import matplotlib.pyplot as plt
import time

mp_pose = mp.solutions.pose

st.set_page_config(page_title="Just Graph", layout="centered")

# workings
if "page" not in st.session_state:
    st.session_state.page = "home"

if "graph_start_timer" not in st.session_state:
    st.session_state.graph_start_time = None

if "cam_start_timer" not in st.session_state:
    st.session_state.cam_start_time = None

if "page" not in st.session_state:
    st.session_state.page = "home"

# new page function
def change_page(new_page):
    st.session_state.page = new_page

# home page
def home():
    st.title("Just Graph")
    st.write("Welcome to Just Graph! This app will test your memory and arm flexibility skills under pressure.")
    st.write("by Sara Koka")

    if st.button("Start"):
        st.session_state.graphs_start_time = time.time()
        change_page("graph")
 
# define graphs
def make_graph_image(graph_type: str):
    # set up the graphs
    st.title("Memorize the Shape of the Graph")

    x = np.linspace(-15, 15, 100)  # 100 points from -15 to 15

    if graph_type == "quadratic":
        y = 1 * x**2 + 2 * x + 1 #quadratic func
    elif graph_type == "horizontal":
        y = np.ones_like(x) * 5 # horizontal line at y=5
    elif graph_type == "increasing":
        y = 0.5 * x + 2 # increasing line with slope 0.5 and intercept 2
    else:
        y = np.zeros_like(x) # default to a horizontal line at y=0

    plt.plot(x, y, label="quadratic: y = 1x^2 + 2x + 1")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)

# graph page
def graph():
    make_graph_image("quadratic")

    graph_duration = 5
    elapsed = time.time() - st.session_state.graphs_start_time
    remaining = max(0, graph_duration - elapsed)

    st.write(f"You have {remaining:.1f} seconds to memorize the graph.")
    if remaining <= 0.0:
        st.session_state.cam_start_time = time.time()
        change_page("camera")
    else:
        time.sleep(1)  # Update every second
        st.rerun()  # Rerun the app to update the timer

# camera page

def camera():
    st.title("Recreate the Graph with Your Arms!")
    st.write("Use your arms to mimic the shape of the graph. The camera will capture your pose and grade you on your accuracy hehe")

    camera_duration = 12
    elapsed = time.time() - st.session_state.cam_start_time
    remaining = max(0, camera_duration - elapsed)

    st.write(f"You have {remaining:.if} seconds")
    if remaining <= 0:
        st.write("Time's up! Let's see how you did!")
        # Here you would add code to analyze the captured pose and grade the user
    else:
        time.sleep(1)  # Update every second
        st.rerun()  # Rerun the app to update the timer
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose()
    # Start video capture
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw pose landmarks on the image
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the resulting image
        st.image(image)



# accuracy page
def accuracy():
    st.title("How'd you do??")

    if st.button("retry"):
        st.session_state.graph.start_time = None
        st.session_state.cam_start_time = None
        change_page("home")

# page routing
if st.session_state.page == "home":
    home()
elif st.session_state.page == "graph":
    graph()
elif st.session_state.page == "camera":
    camera()
elif st.session_state.page == "accuracy":
    accuracy()
