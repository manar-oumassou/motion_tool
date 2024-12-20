import matplotlib.pyplot as plt
import numpy as np

def plot_quiver(flow, roi=None, step=16):
    h, w = flow.shape[:2]
    if roi:
        x, y, w, h = roi
        flow = flow[y:y+h, x:x+w]

    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    plt.figure(figsize=(10, 6))
    plt.quiver(x, y, fx, fy, angles='xy', scale_units='xy', scale=1, color='b')
    plt.gca().invert_yaxis()
    plt.title("Optical Flow Quiver Plot")
    plt.xlabel("X-axis (px)")
    plt.ylabel("Y-axis (px)")
    plt.show()

def plot_profile(profile_data):
    plt.figure(figsize=(10, 4))
    plt.plot(profile_data, label="Velocity Magnitude")
    plt.title("Velocity Profile")
    plt.xlabel("Position")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.show()
