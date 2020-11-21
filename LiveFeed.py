import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime as dt
import time
import os
import tkinter as tk

# Initialize GUI Slider from 0 to 255
gui_loop = tk.Tk()
gui_scale = tk.Scale(gui_loop, from_=0, to=255, orient=tk.HORIZONTAL)
gui_scale.set(20)
gui_scale.pack()

# Initialize camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

# Load Background
bkg = cv2.imread("Background2.png")

try:
    while(True):

        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        gui_loop.update()
        gui_loop.update_idletasks()
        threshold_value = gui_scale.get()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        #bkg_removed = cv2.absdiff(color_image, bkg)
        #cv2.imshow('Real Sense', color_image)
        diff = cv2.absdiff(color_image, bkg)
        #diff = cv2.blur(diff, (9, 9))
        cv2.imshow('Background Difference', diff)

        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply a Gaussian filter to reduce image noise
        blur = cv2.GaussianBlur(diff_gray,(9,9),0)

        ret,thresh1 = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)  # 30?
        
        cv2.imshow("threshold", thresh1)

        #edges = cv2.Canny(thresh1, 400,  500)
        #cv2.imshow("edges", edges)

        #cv2.imshow('RealSense', cv2.Canny(color_image, 200, 250))  # 400/550

        # detect circles in the image
        circles = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 1, 36, param1=100, param2=7, minRadius = 20, maxRadius = 25)


        # ensure at least some circles were found
        if circles is not None:
            print("active")
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(diff, (x, y), r, (0, 0, 255), 4)
                cv2.rectangle(diff, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
            # show the output image
            cv2.imshow("output", diff)

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()