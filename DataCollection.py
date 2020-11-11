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

# Define parameters to collect data
image_count = 5  # Number of images to save
last_time = time.time()  # timer variable for image capturing
time_step = 10  # How long to wait between image captures
save_path = os.getcwd() + "/test_images/"
num_count = 3
try:
    while image_count != 0:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', cv2.medianBlur(color_image, 25))
        cv2.imshow('RealSense', color_image)

        cv2.waitKey(1)

        if (time.time()-last_time) > time_step:
            print("Click!")
            cv2.imwrite(save_path + str(dt.now()) + "__" + str(image_count) + ".png", color_image)
            image_count -= 1
            last_time = time.time()
            num_count = 3

        elif (time.time()-last_time) > time_step - 1 and num_count == 1:
            print(1)
            num_count = 0
        
        elif (time.time()-last_time) > time_step - 2 and num_count == 2:
            print(2)
            num_count = 1

        elif (time.time()-last_time) > time_step - 3 and num_count == 3:
            print(3)
            num_count = 2

finally:

    # Stop streaming
    pipeline.stop()