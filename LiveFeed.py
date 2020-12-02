import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime as dt
import time
import os
import tkinter as tk

from BackgroundImageHandler import BackgroundImageHandler
from ObjectClassifier import ObjectClassifier


#### Initialize Custom Classes ####
BkgHandler = BackgroundImageHandler(30, 0.1)
ObjClassifier = ObjectClassifier()

#### Initialize GUI ####
main_window = tk.Tk()
main_window.geometry('150x300')
main_window.title('8 Ball')

gui_scale = tk.Scale(main_window, from_=0, to=255, orient=tk.HORIZONTAL, length=150, label="Ball Threshold")
gui_scale.set(10)
gui_scale.pack(side=tk.BOTTOM)

table_threshold_scale = tk.Scale(main_window, from_=0, to=255, orient=tk.HORIZONTAL, length=150, label="Table Threshold")
table_threshold_scale.set(10)
table_threshold_scale.pack(side=tk.BOTTOM)

def capture_background():
    BkgHandler.reset_bkg(30, 0.01)
    return

bkg_btn = tk.Button(main_window, text="Capture Background", command=capture_background, width=150)
bkg_btn.pack()

load_bkg_btn = tk.Button(main_window, text="Load Background", command=lambda: BkgHandler.load_background("Background.png"), width=150)
load_bkg_btn.pack()

save_bkg_btn = tk.Button(main_window, text="Save Background", command=lambda: BkgHandler.save_background("Background.png"), width=150)
save_bkg_btn.pack()

bkg_debug_toggle_btn = tk.Button(main_window, text="Debug Toggle", command=BkgHandler.toggle_table_debug, width=150)
bkg_debug_toggle_btn.pack()

exit_btn = tk.Button(main_window, text="Exit", command=main_window.destroy, width=150)
exit_btn.pack()

#### Initialize Camera ####
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
        
        # Skip Loop iteration if we don't have a frame
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        #### Update GUI ####
        main_window.update()
        main_window.update_idletasks()
        ball_threshold = gui_scale.get()

        #### Capture Static Background ####
        if BkgHandler.get_bkg_state() == False:
            BkgHandler.img_accumulator(color_image)
            continue

        rect = BkgHandler.get_table_border()

        #bkg_removed = cv2.absdiff(color_image, bkg)
        cv2.imshow('Real Sense', color_image)
        img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) 
        ret,thresh1 = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)  # 16?
        #blur = cv.GaussianBlur(img_gray,(5,5),0)  #Create a gaussian blur to remove noise and pass through Otsu thresholding
        #ret,thresh1 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #thresh1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2) #alternatively use an adaptive threshold with gaussian noise reduction

        #cv2.imshow("Background", BkgHandler.get_bkg_img())

        
        
        
        # At this point, we have a background image and a border, yea?
        # Next step is to subtract background image and restrict search area, then look for balls and cues
        
        difference_image = cv2.absdiff(color_image, BkgHandler.get_bkg_img())
        #cv2.imshow('Background Difference', difference_image)
        

        ObjClassifier.preprocess_for_scan(difference_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]], ball_threshold)
        contours = ObjClassifier.scan_for_keypoints()

        cv2.imshow("Contours", contours)

        """
        Convert to Grayscale
        Look at difference feed
        (Optional) Apply Gaussian
        Threshold to highlight balls and cue (split)
        
        #### Balls ####
        Open-Close
        (Optional) Contours and Edge Detection?
        (IDK) Partition into segments
        Apply Min Enclosing Circle / Hough Circles with new thresholding

        #### Cue ####
        """
        
        
        
        
        
        
        
        
        
        
        
        """
        # Apply a Gaussian filter to reduce image noise
        blur = cv2.GaussianBlur(diff_gray,(5,5),0)

        ret,thresh1 = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)  # 30?
        
        cv2.imshow("threshold", thresh1)

        #edges = cv2.Canny(thresh1, 400,  500)
        #cv2.imshow("edges", edges)

        #cv2.imshow('RealSense', cv2.Canny(color_image, 200, 250))  # 400/550

        # detect circles in the image
        DrawCircles(thresh, diff)
        """

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()