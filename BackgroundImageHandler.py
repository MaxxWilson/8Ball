"""Defines a class for handling the background of the pool table. Called when we want to recapture a background image for
background removal, averaging the video feed over a time period to reduce noise level"""

import time

import numpy as np
import cv2


class BackgroundImageHandler():
    def __init__(self, img_count = 30, timer = 0.01):
        self.reset_bkg(img_count, timer)

        self._bkg_img_gray = None
        self._bkg_img_thresh = None

        self.bounding_rect = []

        self.debug_toggle

    def img_accumulator(self, img):
        
        if self._avg_counter == self._img_count:
            self._bkg_img = img.astype(np.float32, copy=False)
            self._avg_counter -= 1

        elif self._avg_counter > 0 and (time.time() - self._last_frame_time) > self._snapshot_timer:
            self._last_frame_time = time.time()
            self._avg_counter -= 1
            self._bkg_img += img.astype(np.float32, copy=False)
            
        elif self._avg_counter == 0:
            self._bkg_img = (self._bkg_img / self._img_count).astype("uint8", copy=False)
            self._bkg_state = True

    def calculate_table_border(self, threshold):
        
        self._bkg_img_gray = cv2.cvtColor(self._bkg_img, cv2.COLOR_BGR2GRAY)
        _, self._bkg_img_thresh = cv2.threshold(self._bkg_img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
        self.bounding_rect = cv2.boundingRect(self._bkg_img_thresh)
        
        return self.bounding_rect

    def reset_bkg(self, img_count, timer):
        self._bkg_state = False
        self._bkg_img = None

        self._img_count = img_count
        self._avg_counter = img_count
        
        self._snapshot_timer = timer
        self._last_frame_time = 0

    def toggle_table_debug(self):
        self.debug_toggle = not self.debug_toggle

    def load_background(self, file):
        self._bkg_img = cv2.imread(file)

    def get_bkg_img(self):
        return self._bkg_img

    def get_bkg_state(self):
        return self._bkg_state


# Test for calculate_table_border()
"""
BkgHandler = BackgroundImageHandler()
BkgHandler.load_background("BackgroundAvg.png")
cv2.imshow("Background", BkgHandler.get_bkg_img())
cv2.waitKey(0)
cv2.destroyAllWindows()

print(BkgHandler.calculate_table_border(10))

cv2.imshow("Thresholded Image", BkgHandler._bkg_img_thresh)

rect = cv2.rectangle(BkgHandler.get_bkg_img().copy(),(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Bounds", rect)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""