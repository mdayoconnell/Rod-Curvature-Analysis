#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:03:49 2025

@author: micah
"""

import cv2

video_path = "/Users/micah/Desktop/Vid1.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video")
else:
    print("Video opened successfully")