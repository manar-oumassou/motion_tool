import streamlit as st
import cv2
import os
import pandas as pd
import numpy as np
from src.video_processor import process_optical_flow, extract_velocity_in_roi, analyze_velocity_profile
from src.visualization import plot_quiver, plot_profile
from src.utils import select_roi

# Streamlit App Title
st.title("Optical Flow Analysis Tool with ROI Selection")

# Initialize session state for ROI
if "roi" not in st.session_state:
    st.session_state["roi"] = None

# Upload video
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_video:
    video_path = os.path.join("data/videos", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Process the video
    if st.button("Run Optical Flow Analysis"):
        st.write("Processing video for optical flow...")
        velocity_table, flow_data = process_optical_flow(video_path)
        st.session_state["flow_data"] = flow_data
        st.write("Velocity data processed successfully!")

        # Display Quiver Plot for the full frame
        st.write("### Quiver Plot for Full Frame")
        plot_quiver(flow_data)

    # Select ROI
    st.write("### Select ROI")
    if st.button("Select ROI"):
        # Select ROI only once and store it in session state
        st.session_state["roi"] = select_roi(video_path)
        st.success(f"ROI Selected: {st.session_state['roi']}")

    # If ROI is already selected
    if st.session_state["roi"]:
        roi = st.session_state["roi"]
        st.write(f"Selected ROI: {roi}")

        # Analyze velocity in ROI
        magnitude, angle, avg_velocity = extract_velocity_in_roi(st.session_state["flow_data"], roi)
        st.write(f"Average Velocity in ROI: {avg_velocity:.2f} m/s")
        st.write("Quiver Plot for ROI")
        plot_quiver(st.session_state["flow_data"], roi=roi)

        # Profile Analysis
        st.write("### Profile Analysis")
        profile_line = st.slider("Select Profile Line (Y-Axis Index):", 0, magnitude.shape[0] - 1)
        profile_data = analyze_velocity_profile(magnitude, profile_line, axis="horizontal")
        st.write("Velocity Profile Along Line")
        plot_profile(profile_data)
