import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime as dt
import time
import os

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

pipeline.start(config)

# Load Background
bkg = cv2.imread("1.png")

try:
    while(True):

    # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        #bkg_removed = cv2.absdiff(color_image, bkg)
        cv2.imshow('Real Sense', color_image)

        diff = cv2.absdiff(color_image, bkg)
        #diff = cv2.blur(diff, (9, 9))
        cv2.imshow('Background Difference', diff)

        img_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) 
        ret,thresh1 = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)  # 30?
        cv2.imshow("threshold", thresh1)

        #edges = cv2.Canny(thresh1, 400,  500)
        #cv2.imshow("edges", edges)

        #cv2.imshow('RealSense', cv2.Canny(color_image, 200, 250))  # 400/550

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()