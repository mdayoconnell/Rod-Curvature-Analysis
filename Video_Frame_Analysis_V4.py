#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video curvature analysis tool: tracks arc length of max curvature over time
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/physics/Desktop/Make_Skeleton.py') 
from Make_Skeleton import extract_skeleton_from_image
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d, convolve
from skimage.graph import route_through_array
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import heapq
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score

# Create output directory to put our video frames
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)


# Identifying endpoints

def find_endpoints(skel):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])  ## This isearches across the image and assigns values to a 3x3 array
    convolved = convolve(skel.astype(int), kernel, mode='constant')
    return np.argwhere(convolved == 11) # And the condition for an endpoint (one neighbour) is having 10+1 = 11


def get_max_curvature_from_skeleton(skeleton, time_in_seconds=0.0, show_plot=True):
    if np.mean(skeleton) > 0.5:
        skeleton = ~skeleton

    endpoints = find_endpoints(skeleton)
    if len(endpoints) < 2:
        raise ValueError("Not enough endpoints found")

    start = tuple(endpoints[0])
    end = tuple(endpoints[-1])

    cost = np.where(skeleton, 1.0, 1e6)
    path, _ = route_through_array(cost, start, end, fully_connected=True)
    path = np.array(path)
    y, x = path[:, 0], path[:, 1]

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

    ds = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
    arc_length = np.concatenate([[0], np.cumsum(ds)])
    s_max = arc_length[np.argmax(curvature)]
    s_cm = np.sum(arc_length * curvature) / np.sum(curvature)
    s_total = arc_length[-1]

    if show_plot:
        max_idx = np.argmax(curvature)
        x_max, y_max = x_smooth[max_idx], y_smooth[max_idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(~skeleton, cmap='gray', origin='lower')  # white background, black rod
        sc = plt.scatter(x_smooth, y_smooth, c=curvature, cmap='hot', s=5)
        plt.plot(x_smooth[0], y_smooth[0], 'bo', label='Start')
        plt.plot(x_smooth[-1], y_smooth[-1], 'go', label='End')
        plt.plot(x_max, y_max, 'ro', label='Max Curvature', markersize=8)
        plt.title(f"Curvature Overlay\nTime = {time_in_seconds:.5f}s, "
          f"s_max = {s_max:.1f}px, Max = {np.max(curvature):.5f}")        
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.colorbar(sc, label="Curvature")
        plt.legend()
        plt.tight_layout()
        plt.show()
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
        
    return s_max, s_cm, s_total, max(curvature)


        
def manual_fit_segments(df_clean, y_column="L_max"):
    import matplotlib.ticker as ticker
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    while True:
        print(f"\nFitting {y_column} vs √t")
        num_fits = int(input("How many linear segments do you want to fit? "))
        fits = []

        for i in range(num_fits):
            start = float(input(f"Start of fit {i+1} (in √t units): "))
            end = float(input(f"End of fit {i+1} (in √t units): "))

            mask = (df_clean["sqrt_time"] >= start) & (df_clean["sqrt_time"] <= end)
            x = df_clean.loc[mask, "sqrt_time"].values.reshape(-1, 1)
            y = df_clean.loc[mask, y_column].values / 100  # convert to meters

            if len(x) < 2:
                print(f"Not enough points for fit {i+1}, skipping.")
                continue

            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x)
            r2 = r2_score(y, y_pred)
            slope = model.coef_[0]
            fits.append((x.flatten(), y_pred * 100, slope, r2))  # convert back to cm

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(df_clean["sqrt_time"], df_clean[y_column], 'r-', label=f"{y_column} (cm)")
        colors = ["orange", "green", "red", "purple", "blue", "brown"]

        for i, (x, y_fit, slope, r2) in enumerate(fits):
            plt.plot(x, y_fit, '-', label=f"α{i+1} ≈ {slope:.2f} m/√s, $R^2$ = {r2:.3f}", color=colors[i % len(colors)])

        plt.xlabel("√Time (s$^{0.5}$)")
        plt.ylabel(f"{y_column} (cm)")
        plt.title(f"Manual Fit: {y_column} vs √t")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.02))
        plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.005))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        plt.tick_params(axis='both', which='major', labelsize=9)
        plt.tick_params(axis='both', which='minor', labelsize=0)
        plt.legend()
        plt.tight_layout()
        plt.show()

        satisfied = input("Are you satisfied with the fits? (y/n): ").strip().lower()
        if satisfied == "y":
            break

    


def process_video(video_path, fps, data, real_length_cm=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video")

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_number / fps
        try:
            tmp_path = os.path.join(output_dir, f"frame_{frame_number}.png")
            cv2.imwrite(tmp_path, frame)
            skeleton = extract_skeleton_from_image(tmp_path) # If you want to change the threshhold (keep x% brightest) change that in Make_Skeleton
            s_max, s_cm, s_total, k_max = get_max_curvature_from_skeleton(skeleton, time_in_seconds=time_sec, show_plot=False)

            entry = {
                "frame": frame_number,
                "time": time_sec,
                "sqrt_time": np.sqrt(time_sec),
                "s_max": s_max,
                "s_cm": s_cm,
                "s_total": s_total,
                "maximum curvature": k_max,
                "L_max": (s_max/s_total)*real_length_cm
                    
            }
            if real_length_cm is not None:
                entry["s_max_real"] = s_max / s_total * real_length_cm
                entry["L_real"] = real_length_cm

            data.append(entry)

        except Exception as e:
            print(f"Error in frame {frame_number}: {e}")
            
        
        frame_number += 1

    cap.release()
    return data


def mark_and_filter_outliers(df):
    df = df.copy()

    curv_thresh = 2 * df["maximum curvature"].median()
    total_thresh = 0.95 * df["s_total"].median()

    df["is_outlier"] = (
        (df["maximum curvature"] >= curv_thresh) |
        (df["s_total"] < total_thresh) |
        (df["s_max"] < 100)
    )

    df_clean = df[~df["is_outlier"]].reset_index(drop=True)  # <-- Move this up

    df_clean["L_max"] = df_clean["s_max"] / df_clean["s_total"] * df_clean["L_real"]
    df_clean["L_cm"] = df_clean["s_cm"] / df_clean["s_total"] * df_clean["L_real"]


    return df, df_clean


def closeout(df):
    if df.empty:
        print("No data to process.")
        return

    df, df_clean = mark_and_filter_outliers(df)

    # --- Plot s_max vs sqrt(time) (with outliers marked) ---
    plt.figure(figsize=(8, 4))
    for is_outlier, group in df.groupby("is_outlier"):
        label = "Outlier" if is_outlier else "Inlier"
        color = "green" if is_outlier else "blue"
        plt.plot(group["sqrt_time"], group["s_max"], 'o', label=label, color=color)

    plt.xlabel("√Time (s^0.5)")
    plt.ylabel("Arc Length of Max Curvature (px)")
    plt.title("Fracture Tip Propagation (with Outliers Marked)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot max curvature vs time (with outliers marked) ---
    plt.figure(figsize=(8, 4))
    for is_outlier, group in df.groupby("is_outlier"):
        label = "Outlier" if is_outlier else "Inlier"
        color = "green" if is_outlier else "red"
        plt.plot(group["time"], group["maximum curvature"], 'o', label=label, color=color)

    plt.xlabel("Time (s)")
    plt.ylabel("Maximum Curvature")
    plt.title("Maximum Curvature vs Time (with Outliers Marked)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Replot with outliers removed ---
    print(f"Removed {len(df) - len(df_clean)} outliers.\n")

    plt.figure(figsize=(8, 4))
    plt.plot(df_clean["sqrt_time"], df_clean["s_max"], 'o-', label="s_max vs √t")
    plt.xlabel("√Time (s^0.5)")
    plt.ylabel("Arc Length of Max Curvature (px)")
    plt.title("Fracture Tip Propagation (Outliers Removed)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Compute ratio of max curvature to initial curvature ---
    initial_curv = df_clean["maximum curvature"].iloc[0]
    max_curv = df_clean["maximum curvature"].max()
    ratio = max_curv / initial_curv if initial_curv != 0 else np.nan

    plt.figure(figsize=(8, 4))
    plt.plot(df_clean["time"], df_clean["maximum curvature"], 'o-', label="Max curvature")
    plt.xlabel("Time (s)")
    plt.ylabel("Maximum Curvature")
    plt.title(f"Max Curvature vs Time (Outliers Removed)\nMax/Initial = {ratio:.2f}")    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.plot(df_clean["sqrt_time"], df_clean["s_max"], 'o-', label="s_max", alpha=0.7)
    plt.plot(df_clean["sqrt_time"], df_clean["s_cm"], 'x-', label="s_cm", alpha=0.7)
    plt.xlabel("√Time (s^0.5)")
    plt.ylabel("Arc Length (px)")
    plt.title("s_max vs s_cm (Outliers Removed)") #S_max is the point of max curvature, s_cm takes an integral to find the "center of mass" of the curvature wave
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Save filtered data ---
    np.savez("curvature_arrays_filtered.npz",
             time=df_clean["time"].values,
             sqrt_time=df_clean["sqrt_time"].values,
             s_max=df_clean["s_max"].values,
             s_total=df_clean["s_total"].values,
             s_max_real=df_clean.get("s_max_real", pd.Series()).values,
             L_real=df_clean.get("L_real", pd.Series()).values)

    df_clean.to_csv("curvature_analysis_output_filtered.csv", index=False)
    print("Saved filtered CSV and NPZ.")
    
    # --- Final Plot to help us 
    print('brooooo')
    plt.figure(figsize=(8, 4))
    plt.plot(df_clean["sqrt_time"], df_clean["L_max"], 'o-', label="L_max (cm)")
    
    plt.xlabel("√Time (s$^{0.5}$)")
    plt.ylabel("L_max (cm)")
    plt.title("Fracture Tip Position vs √t")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Tighter but less intrusive grid
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.02))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    
    # Hide some tick labels to reduce clutter
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.tick_params(axis='both', which='minor', labelsize=0)
    plt.show()
    
    fit_max = input("Would you like to fit L_max (from s_max)? [y/n]: ").strip().lower()
    if fit_max == "y":
        manual_fit_segments(df_clean, y_column="L_max")

    fit_cm = input("Would you like to fit L_cm (from s_cm)? [y/n]: ").strip().lower()
    if fit_cm == "y":
        manual_fit_segments(df_clean, y_column="L_cm")


    
if __name__ == "__main__":
    video_path = "/Users/micah/Desktop/Vid24.mov"
    fps = 5000
    real_length_cm = 24.5

    result_data = []

    try:
        process_video(video_path, fps, data=result_data, real_length_cm=real_length_cm)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received — exporting data collected so far.")
    finally:
        df = pd.DataFrame(result_data)
        closeout(df)
        
        '''
        subset = df[(df["sqrt_time"] > 0.08) & (df["sqrt_time"] < 0.12)]
        plt.plot(subset["sqrt_time"], subset["s_max"], 'o-')
        plt.xlabel("√Time")
        plt.ylabel("s_max")
        plt.title("Suspicious Region")
        plt.show()
        '''

