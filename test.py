import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Webcam Test")

# Basic webcam stream without transformations
webrtc_streamer(key="simple-webcam")