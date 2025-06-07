#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:44:39 2025

@author: micah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video curvature analysis tool: tracks arc length of max curvature over time
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Make_Skeleton import extract_skeleton_from_image
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d, convolve
from skimage.graph import route_through_array
import pandas as pd


def find_endpoints(skel):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    convolved = convolve(skel.astype(int), kernel, mode='constant')
    return np.argwhere(convolved == 11)


def get_max_curvature_from_skeleton(skeleton, time_in_seconds=0.0, show_plot=True):
    if np.mean(skeleton) > 0.5:
        skeleton = ~skeleton
    endpoints = find_endpoints(skeleton)
    start = tuple(endpoints[0])
    end = tuple(endpoints[1])
    if len(endpoints) < 2:
        raise ValueError("Not enough endpoints found")

    # Sort endpoints to enforce consistent direction — e.g., top-to-bottom or left-to-right
    endpoints = sorted(endpoints, key=lambda pt: pt[1])  # sort by y-coordinate (row), for vertical rods
    # If your rod is more horizontal, use key=lambda pt: pt[1] instead

    start = tuple(endpoints[0])
    end = tuple(endpoints[-1])

    cost = np.where(skeleton, 1.0, 1e6)
    path, _ = route_through_array(cost, start, end, fully_connected=True)
    path = np.array(path)
    y, x = path[:, 0], path[:, 1]
    
    if path[0, 0] > path[-1, 0]:  # adjust this check based on geometry
        path = path[::-1]  # reverse path
    

    tck, _ = splprep([x, y], s=300, k=3)
    u_new = np.linspace(0, 1, 500)
    x_smooth, y_smooth = splev(u_new, tck)

    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2)**1.5
    denom[denom == 0] = 1e-6
    curvature = np.abs((dx * ddy - dy * ddx) / denom)
    curvature = gaussian_filter1d(curvature, sigma=5)
    
    # Smooth curvature

    # Cut off both ends
    cut_head = 35
    cut_tail = 10
    curvature_subset = curvature[cut_head:-cut_tail]
    x_subset = x_smooth[cut_head:-cut_tail]
    y_subset = y_smooth[cut_head:-cut_tail]
    
    # Arc length and corrected s_max
    ds = np.sqrt(np.diff(x_subset)**2 + np.diff(y_subset)**2)
    arc_length = np.concatenate([[0], np.cumsum(ds)])
    s_total = arc_length[-1]
    
    max_idx_subset = np.argmax(curvature_subset)
    true_idx = max_idx_subset
    s_max = arc_length[true_idx]
    
    print(s_max, max_idx_subset)
    
    if show_plot:
        x_max, y_max = x_subset[true_idx], y_subset[true_idx]

    plt.figure(figsize=(6, 6))
    plt.imshow(~skeleton, cmap='gray', origin='lower')
    sc = plt.scatter(x_subset, y_subset, c=curvature_subset, cmap='hot', s=5)
    #plt.plot(x_subset[0], y_subset[0], 'bo', label='Start')
    #plt.plot(x_subset[-1], y_subset[-1], 'go', label='End')
    plt.plot(x_max, y_max, 'ro', label='Point of Max Curvature', markersize=8)
    plt.title(f"Curvature Heatmap \nTime = {time_in_seconds:.5f}s, Curvature = {np.max(curvature_subset):.5f}")        
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.colorbar(sc, label="Curvature")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    cutoff = 20  # number of points to exclude from end
    curvature_subset = curvature[:-cutoff]
    x_subset = x_smooth[:-cutoff]
    y_subset = y_smooth[:-cutoff]
    
    ds = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
    arc_length = np.concatenate([[0], np.cumsum(ds)])
    s_total = arc_length[-1]

    max_idx_subset = np.argmax(curvature_subset)
    s_max = arc_length[max_idx_subset]
        
    return s_max, s_total, max_idx_subset
'''
        plt.figure(figsize=(8, 4))
        plt.plot(arc_length, curvature)
        plt.xlabel("Arc Length (px)")
        plt.ylabel("Curvature")
        plt.title("Curvature vs Arc Length")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
'''

    



def process_images(image_paths, real_length_cm=25):
    data = []

    for frame_number, path in enumerate(image_paths):
        time_sec = frame_number / 125  # Hardcoded FPS = 5000, adjust if needed

        try:
            skeleton = extract_skeleton_from_image(path, debug=False)
            s_max, s_total, max_curv = get_max_curvature_from_skeleton(skeleton, time_in_seconds=time_sec, show_plot=True)

            entry = {
                "frame": frame_number,
                "time": time_sec,
                "sqrt_time": np.sqrt(time_sec),
                "s_max": s_max,
                "s_total": s_total,
                "max_curvature": max_curv
            }

            if real_length_cm is not None:
                entry["s_max_real"] = s_max / s_total * real_length_cm
                entry["L_real"] = real_length_cm

            data.append(entry)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return data


if __name__ == "__main__":
    image_paths = [
        #"/Users/micah/Desktop/Frame1.png",
        "/Users/micah/Desktop/Frame2.png",
        #"/Users/micah/Desktop/Frame3.png",
        "/Users/micah/Desktop/Frame4.png",
    ]
    real_length_cm = 10.0

    result_data = process_images(image_paths, real_length_cm=real_length_cm)

    if result_data:
        df = pd.DataFrame(result_data)

        plt.figure(figsize=(8, 4))
        plt.plot(df["sqrt_time"], df["s_max"], 'o-', label="s_max vs √t")
        plt.xlabel("√Time (s^0.5)")
        plt.ylabel("Arc Length of Max Curvature (px)")
        plt.title("Fracture Tip Propagation\ns_max vs √t")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        df.to_csv("curvature_analysis_output.csv", index=False)
        print("Saved CSV with curvature data.")
    else:
        print("No curvature data was extracted.")