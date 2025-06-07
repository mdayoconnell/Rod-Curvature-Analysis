import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d, convolve
from skimage.morphology import skeletonize
from skimage.graph import route_through_array


def find_endpoints(skel):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    convolved = convolve(skel.astype(int), kernel, mode='constant')
    endpoints = np.argwhere(convolved == 1)
    return endpoints


def get_max_curvature_from_frame(frame, time_in_seconds=0.0, show_plot=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (7, 7), 0)

    # Threshold to extract darkest 10% of pixels
    hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
    cumulative = np.cumsum(hist)
    total_pixels = blurred.shape[0] * blurred.shape[1]
    threshold_val = np.argmax(cumulative > total_pixels * 0.1)
    _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

    # Skeletonize
    skeleton = skeletonize(binary > 0)

    # Find endpoints
    endpoints = find_endpoints(skeleton)
    if len(endpoints) < 2:
        raise ValueError("Not enough endpoints found")

    start = tuple(endpoints[0])
    end = tuple(endpoints[-1])

    # Route through skeleton
    cost = np.where(skeleton, 1.0, 1e6)
    path, _ = route_through_array(cost, start, end, fully_connected=True)
    path = np.array(path)
    y, x = path[:, 0], path[:, 1]

    # Fit spline
    tck, _ = splprep([x, y], s=300, k=3)
    u = np.linspace(0, 1, 500)
    x_smooth, y_smooth = splev(u, tck)

    # Compute curvature
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5
    denom[denom == 0] = 1e-6
    curvature = np.abs((dx * ddy - dy * ddx) / denom)
    curvature = gaussian_filter1d(curvature, sigma=5)

    # Arc length
    ds = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
    arc_length = np.concatenate([[0], np.cumsum(ds)])
    s_max = arc_length[np.argmax(curvature)]
    s_total = arc_length[-1]

    if show_plot:
        idx_max = np.argmax(curvature)
        x_max, y_max = x_smooth[idx_max], y_smooth[idx_max]

        plt.figure(figsize=(6, 6))
        plt.imshow(skeleton, cmap='gray_r')  # Inverted colormap for white background
        sc = plt.scatter(x_smooth, y_smooth, c=curvature, cmap='hot', s=5)
        plt.plot(x_smooth[0], y_smooth[0], 'bo', label='Start')
        plt.plot(x_smooth[-1], y_smooth[-1], 'go', label='End')
        plt.plot(x_max, y_max, 'ro', label='Max Curvature', markersize=8)
        plt.title(f"Curvature Heatmap Overlay\nTime = {time_in_seconds:.2f}s, s_max = {s_max:.1f}px")
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.colorbar(sc, label="Curvature")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Curvature vs Arc length
        plt.figure(figsize=(8, 4))
        plt.plot(arc_length, curvature, color='black')
        plt.xlabel("Arc Length (px)")
        plt.ylabel("Curvature")
        plt.title("Curvature vs Arc Length")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return s_max, s_total


if __name__ == "__main__":
    video_path = "/Users/micah/Desktop/Vids/Vid24.mov"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    get_max_curvature_from_frame(frame, time_in_seconds=0.0, show_plot=True)
    cap.release()