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
main_window.geometry('150x400')
main_window.title('8 Ball')

# Scale for Ball Thresholding
gui_scale = tk.Scale(main_window, from_=0, to=255, orient=tk.HORIZONTAL, length=150, label="Ball Threshold")
gui_scale.set(10)
gui_scale.pack(side=tk.BOTTOM)

# Scale for Table Thresholding
table_threshold_scale = tk.Scale(main_window, from_=0, to=255, orient=tk.HORIZONTAL, length=150, label="Table Threshold")
table_threshold_scale.set(10)
table_threshold_scale.pack(side=tk.BOTTOM)

def capture_background():
    BkgHandler.reset_bkg(30, 0.01)
    return

# Button to capture a new background from Background Handler
bkg_btn = tk.Button(main_window, text="Capture Background", command=capture_background, width=150)
bkg_btn.pack()

# Button to load a previous background image
load_bkg_btn = tk.Button(main_window, text="Load Background", command=lambda: BkgHandler.load_background("Background.png"), width=150)
load_bkg_btn.pack()

# Button to save new background image
save_bkg_btn = tk.Button(main_window, text="Save Background", command=lambda: BkgHandler.save_background("Background.png"), width=150)
save_bkg_btn.pack()

# Button to toggle debug options for Background Handler, primarily table thresholding
bkg_debug_toggle_btn = tk.Button(main_window, text="Debug Toggle", command=BkgHandler.toggle_table_debug, width=150)
bkg_debug_toggle_btn.pack()

# Button to save images of search regions from Object Classifier
save_search_regions_btn = tk.Button(main_window, text="Save Regions", command=ObjClassifier.save_regions, width=150)
save_search_regions_btn.pack()

def save_output():
    cv2.imwrite("Circles.png", ObjClassifier.diff_img)

# Button to save final output
fin_save_btn = tk.Button(main_window, text="Save Output", command=save_output, width=150)
fin_save_btn.pack()

# Button to abort GUI Loop and exit program
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
        table_threshold = table_threshold_scale.get()

        #### Capture Frame Averaged Background ####
        if BkgHandler.get_bkg_state() == False:
            BkgHandler.img_accumulator(color_image)
            continue

        #### Debug Options for Background Handler ####
        if BkgHandler.debug_toggle:
            BkgHandler.calculate_table_border(table_threshold)
            tbl_rgn = BkgHandler.get_table_border()
            table_bounds_img = cv2.rectangle(BkgHandler.get_bkg_img().copy(),(tbl_rgn[0][0],tbl_rgn[0][1]),(tbl_rgn[1][0],tbl_rgn[1][1]),(0,255,0),2)
            
            cv2.imshow("Table Border", table_bounds_img)
            cv2.imshow("Table Border Threshold Image", BkgHandler._bkg_img_thresh)
        
        else:
            tbl_rgn = BkgHandler.get_table_border()

        # Calculate Difference Image
        difference_image = cv2.absdiff(color_image, BkgHandler.get_bkg_img())
        
        # Pass restriced table region into Object Classifier for smoothing and binarization
        ObjClassifier.preprocess_for_scan(color_image[tbl_rgn[0][1]:tbl_rgn[1][1], tbl_rgn[0][0]:tbl_rgn[1][0]], difference_image[tbl_rgn[0][1]:tbl_rgn[1][1], tbl_rgn[0][0]:tbl_rgn[1][0]], ball_threshold)
        cv2.imshow("Binary Image", ObjClassifier.binary_img)
        
        ObjClassifier.scan_for_key_regions() # Identify key regions for object detection
        ObjClassifier.identify_objects() # Classify balls and cues 

        contours = ObjClassifier.draw_search_regions()
        #circles = ObjClassifier.draw_circles()


        cv2.imshow("Contours", contours)
        #cv2.imshow("Circles", circles)

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()