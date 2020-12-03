import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime as dt
import time
import sys
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

        #### TEST PLEASE ####
        if BkgHandler.debug_toggle:
            rect = BkgHandler.get_table_border()
            table_bounds_img = cv2.rectangle(BkgHandler.get_bkg_img().copy(),(rect[0][0],rect[0][1]),(rect[1][0],rect[1][1]),(0,255,0),2)
            cv2.imshow("Table Border", table_bounds_img)
            cv2.imshow("Table Border Threshold Image", BkgHandler._bkg_img_thresh)

        #cv2.imshow("Background", BkgHandler.get_bkg_img())

        
    
        
        # At this point, we have a background image and a border, yea?
        # Next step is to subtract background image and restrict search area, then look for balls and cues
        
        difference_image = cv2.absdiff(color_image, BkgHandler.get_bkg_img())
        #cv2.imshow('Background Difference', difference_image)
        

        ObjClassifier.preprocess_for_scan(color_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]], ball_threshold)
        cv2.imshow("Binary Image", ObjClassifier.frame_avg)
        
        #ObjClassifier.scan_for_keypoints()
        #contours = ObjClassifier.draw_search_regions()
        #ObjClassifier.find_balls()
        #circles = ObjClassifier.draw_circles()

        #cv2.imshow("Contours", contours)
        #cv2.imshow("Circles", circles)

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()