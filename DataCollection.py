import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime as dt
import time
import os

# Open video pipeline and configure camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

# Define parameters to collect data
image_directory = "test_images/" # Local path to image folder
save_path = os.getcwd() + "/" + image_directory # Absolute path to image folder

# Look at existing test images for naming scheme
if os.listdir(image_directory) != []:
    name_count = max([int(i.strip(".png")) for i in os.listdir(image_directory)]) + 1
else:
    name_count = 1

print(name_count)

last_time = time.time()  # timer variable for image capturing
time_step = 10  # How long to wait between image captures
image_count = 5 + name_count # Number of images to save
timer_count = 3

try:
    while name_count != image_count:

        # Wait for color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        cv2.waitKey(1)

        if (time.time()-last_time) > time_step:
            print("Click!")
            cv2.imwrite(save_path + str(name_count) + ".png", color_image)
            name_count += 1
            last_time = time.time()
            timer_count = 3

        elif (time.time()-last_time) > time_step - 1 and timer_count == 1:
            print(1)
            timer_count = 0
        
        elif (time.time()-last_time) > time_step - 2 and timer_count == 2:
            print(2)
            timer_count = 1

        elif (time.time()-last_time) > time_step - 3 and timer_count == 3:
            print(3)
            timer_count = 2

finally:

    # Stop streaming
    pipeline.stop()