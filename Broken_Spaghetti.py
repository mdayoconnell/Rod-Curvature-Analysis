#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:18:52 2025

@author: micah
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# --- PCA-based ordering for each contour ---
def order_path_pca(coords):
    coords = coords.astype(float)
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    u, _, _ = np.linalg.svd(centered.T)
    proj = centered @ u[:, 0]
    return coords[np.argsort(proj)]

# --- Load and preprocess image ---
img = cv2.imread('/Users/micah/Desktop/Broken_Spaghetti.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# --- Threshold to isolate dark curves ---
_, binary = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# --- Find contours ---
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# --- Color palette and results storage ---
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
segment_data = []

# --- Plot the fitted curves ---
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Fitted Curves on Broken "Spaghetti"')
plt.axis('equal')
plt.gca().invert_yaxis()

for i, cnt in enumerate(contours):
    if len(cnt) < 30:
        continue

    coords = cnt[:, 0, :]
    coords = np.unique(coords, axis=0)
    if coords.shape[0] < 5:
        continue

    ordered = order_path_pca(coords)
    x = ordered[:, 0]
    y = ordered[:, 1]

    try:
        tck, u = splprep([x, y], s=20, k=3, per=False)
        u_fine = np.linspace(0, 1, 200)
        x_smooth, y_smooth = splev(u_fine, tck)

        # Arc length
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        ds = np.sqrt(dx**2 + dy**2)
        arc_len = np.sum(ds)

        color = colors[i % len(colors)]
        segment_data.append({
            'label': f'S{i+1}',
            'length': arc_len,
            'color': color
        })

        # Plot curve
        plt.plot(x_smooth, y_smooth, color=color, linewidth=2, label=f"Segment {i+1}")

    except Exception as e:
        print(f"Skipping segment {i+1} due to spline error: {e}")
        continue

plt.legend()
plt.tight_layout()
plt.show()

# --- Print arc lengths in image order ---
print("\nSegment arc lengths (pixels):")
for seg in segment_data:
    print(f"{seg['label']}: {seg['length']:.2f}")

# --- Histogram with color-matched bars ---
plt.figure(figsize=(6, 4))
for i, seg in enumerate(segment_data):
    plt.bar(i + 1, seg['length'], color=seg['color'], edgecolor='black')

plt.xticks(range(1, len(segment_data)+1), [seg['label'] for seg in segment_data])
plt.xlabel("Segment")
plt.ylabel("Arc Length (pixels)")
plt.title('Broken "Spaghetti" Segment Lengths')
plt.tight_layout()
plt.show()