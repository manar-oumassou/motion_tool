import cv2
import numpy as np

def process_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    velocity_table = []

    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        if frame_count % 1 == 0:
            if prev_frame is not None:
                flow = calculate_optical_flow(prev_frame, frame)
            prev_frame = frame

        frame_count += 1

    cap.release()
    return None, flow

def calculate_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def extract_velocity_in_roi(flow, roi):
    x, y, w, h = roi
    roi_flow = flow[y:y+h, x:x+w]
    magnitude, angle = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])

    avg_velocity = np.mean(magnitude)
    return magnitude, angle, avg_velocity

def analyze_velocity_profile(magnitude, profile_line, axis="horizontal"):
    if axis == "horizontal":
        profile_data = magnitude[profile_line, :]
    elif axis == "vertical":
        profile_data = magnitude[:, profile_line]
    else:
        raise ValueError("Invalid axis. Choose 'horizontal' or 'vertical'.")
    return profile_data
