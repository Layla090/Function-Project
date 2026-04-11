import streamlit as st
import mediapipe as mp
import cv2
import av
# NumPy is a library for working with arrays and matrices, which are essential for image processing tasks.
import numpy as np
import matplotlib.pyplot as plt
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import VideoProcessorBase

# Mediapipe for live stream
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker.task"

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
        st.rerun()
 
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

    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend()
    plt.grid(True)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    st.pyplot(fig)

# graph page

def graph():
    st.title("Memorize the Shape of the Graph")

    graphs_duration = 5
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

#vid processor
class PoseVideoProcessor(VideoProcessorBase): #processor analyses the photo
    
    def __init__(self):
        options = PoseLandmarkerOptions(
             base_options=BaseOptions(model_asset_path=MODEL_PATH), #use pose model file
             running_mode=VisionRunningMode.VIDEO, #treat like a video
             num_poses=1, # look for one person
             min_pose_detection_confidence=0.5, #confidence that it is said body part
             min_pose_presence_confidence=0.5, #confidence that a person is there
             min_tracking_confidence=0.5, #confidence for movement tracking
        )

        self.landmarker = PoseLandmarker.create_from_options(options) # build the pose landmarker and save it as "self landmarker"

    def recv(self, frame): #recv runs everytime a new vid frame comes in. frame = 1 image from the live video
        img = frame.to_ndarray(format="bgr24") #camera pixels are now data mwahahaha
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert openCV's BGR to mediapipe's RGB for correct color
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) # make the image ready for Mediapipe

        if not hasattr(self, "start_time"):
            self.start_time = time.time()

        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms) # actual pose detection step. Result func. Find body landmarks, result stores them

        if result.pose_landmarks: # if the pose is found then...
            cv2.putText(img, "POSE DETECTED", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            landmarks = result.pose_landmarks[0]
            h, w, _ = img.shape

            for i in [11, 13, 15, 12, 14, 16]: #shoulders, elbows, wrists
                lm = landmarks[i]
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(img, (cx, cy), 10, (255, 105, 180), -1)
        else:
            cv2.putText(img, "NO POSE", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# camera page
def camera():
    st.title("Recreate the Graph with Your Arms!")
    st.write("Use your arms to mimic the shape of the graph.")

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="pose",
        video_processor_factory=PoseVideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    # Initialize timer
    if st.session_state.cam_start_time is None:
        st.session_state.cam_start_time = time.time()
    elapsed = time.time() - st.session_state.cam_start_time
    remaining = max(0, 10 - elapsed)

    st.write(f"Time left: {remaining:.1f} seconds")

    if remaining <= 0:
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