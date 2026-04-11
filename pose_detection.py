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

if "graphs_start_time" not in st.session_state:
    st.session_state.graphs_start_time = None

if "cam_start_time" not in st.session_state:
    st.session_state.cam_start_time = None

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
    st.title("Memorize the Shape of the Graph")

    graphs_duration = 7
    elapsed = time.time() - st.session_state.graphs_start_time
    remaining = max(0, graphs_duration - elapsed)

    st.write(f"You have {remaining:.1f} seconds to memorize the graph.")
    
    make_graph_image("quadratic")

    if remaining == 0:
        st.session_state.graphs_start_time = None
        st.session_state.cam_start_time = time.time()
        change_page("camera")
        st.rerun()  # Rerun the app to immediately switch to the camera page

    else:
        time.sleep(1)  # Update every second
        st.rerun()  # Rerun the app to update the timer

# camera page

def camera():
    st.title("Recreate the Graph with Your Arms!")
    st.write("Use your arms to mimic the shape of the graph.")

    duration = 10

    # Initialize timer
    if st.session_state.cam_start_time is None:
        st.session_state.cam_start_time = time.time()

    # Initialize storage for last frame
    if "last_frame" not in st.session_state:
        st.session_state.last_frame = None

    elapsed = time.time() - st.session_state.cam_start_time
    remaining = max(0, duration - elapsed)

    st.write(f"You have {remaining:.1f} seconds")

    frame_placeholder = st.empty()

    # 🎥 Capture ONE frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Camera not working 😬")
        return

    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    if "pose" not in st.session_state:
        st.session_state.pose = mp_pose.Pose()

    results = st.session_state.pose.process(image)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Always store latest frame
    st.session_state.last_frame = image

    # ⏱ BEFORE TIME ENDS → show live feed
    if remaining > 0:
        frame_placeholder.image(image)
        time.sleep(0.05)
        st.rerun()

    # 🧊 AFTER TIME ENDS → freeze frame
    else:
        st.write("📸 Freeze frame!")
        frame_placeholder.image(st.session_state.last_frame)

        # Move to next page after short delay
        time.sleep(2)
        change_page("accuracy")
        st.rerun()

# accuracy page
def accuracy():
    st.title("How'd you do??")

    if st.button("retry"):
        st.session_state.graphs_start_time = None
        st.session_state.cam_start_time = None
        change_page("home")
        st.rerun()  # Rerun the app to immediately switch to the home page

# page routing
if st.session_state.page == "home":
    home()
elif st.session_state.page == "graph":
    graph()
elif st.session_state.page == "camera":
    camera()
elif st.session_state.page == "accuracy":
    accuracy()