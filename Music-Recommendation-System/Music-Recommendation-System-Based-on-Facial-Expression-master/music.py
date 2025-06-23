import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import os

st.header("Emotion Based Music Recommender")


required_files = {
    "model.h5": "the trained emotion recognition model",
    "labels.npy": "the emotion labels"
}

missing_files = []
for file, description in required_files.items():
    if not os.path.exists(file):
        missing_files.append(f"- {file} ({description})")

if missing_files:
    st.error("Required files are missing!")
    st.write("Please complete the following steps before running this application:")
    st.write("1. Run data_collection.py to collect emotion data (at least 2 emotions)")
    st.write("2. Run data_training.py to train the model")
    st.write("\nMissing files:")
    for file in missing_files:
        st.write(file)
    st.stop()


try:
    model = load_model("model.h5")
    label = np.load("labels.npy")
    st.success("Model and labels loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or labels: {str(e)}")
    st.write("Please ensure you have run the data collection and training steps correctly.")
    st.stop()


holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not(emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            
            try:
                pred = label[np.argmax(model.predict(lst))]
                print(pred)
                cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))
            except Exception as e:
                print(f"Prediction error: {str(e)}")

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)


st.subheader("Music Preferences")
lang = st.text_input("Language")
singer = st.text_input("Singer (optional)")

if lang and st.session_state["run"] != "false":
    st.write("Please look at the camera and show your emotion...")
    webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)


st.sidebar.subheader("Available Emotions")
try:
    available_emotions = list(label)
    st.sidebar.write("The system can recognize these emotions:")
    for emotion in available_emotions:
        st.sidebar.write(f"- {emotion}")
except:
    st.sidebar.write("No emotions loaded")

btn = st.button("Recommend me songs")

if btn:
    if not(emotion):
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        search_query = f"{lang}+{emotion}+song"
        if singer:
            search_query += f"+{singer}"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"