import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime as dt
import time
import sys
import tkinter as tk
from DrawCircles import DrawCircles
from BackgroundImageHandler import BackgroundImageHandler

#### Initialize Background Image Handler ####
BkgHandler = BackgroundImageHandler(30, 0.1)

#### Initialize GUI ####
main_window = tk.Tk()
main_window.geometry('150x300')
main_window.title('8 Ball')

gui_scale = tk.Scale(main_window, from_=0, to=255, orient=tk.HORIZONTAL, length=150, label="Ball Threshold")
gui_scale.set(20)
gui_scale.pack(side=tk.BOTTOM)

def capture_background():
    BkgHandler.reset_bkg(30, 0.01)
    return

bkg_btn = tk.Button(main_window, text="Capture Background", command=capture_background, width=150)
bkg_btn.pack()

exit_btn = tk.Button(main_window, text="Exit", command=main_window.destroy, width=150)
exit_btn.pack()

#### Initialize Camera ####
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

while(True):

    #### Update GUI ####
    main_window.update()
    main_window.update_idletasks()
    threshold_value = gui_scale.get()
    time.sleep(0.001)

try:
    while(False):
        print("Active")
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # Skip Loop iteration if we don't have a frame
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        #### Update GUI ####
        main_window.update()
        main_window.update_idletasks()
        threshold_value = gui_scale.get()

        #### Capture Static Background ####
        if BkgHandler.get_bkg_state() == False:
            print("Takin' Pics!")
            BkgHandler.img_accumulator(color_image)
            continue


        cv2.imshow("Background", BkgHandler.get_bkg_img())
        """
        diff = cv2.absdiff(color_image, )
        cv2.imshow('Background Difference', diff)

        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply a Gaussian filter to reduce image noise
        blur = cv2.GaussianBlur(diff_gray,(5,5),0)

        _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("threshold", thresh)

        # detect circles in the image
        DrawCircles(thresh, diff)
        """

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()