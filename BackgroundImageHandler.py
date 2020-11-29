"""Defines a class for handling the background of the pool table. Called when we want to recapture a background image for
background removal, averaging the video feed over a time period to reduce noise level"""

import numpy as np
import time

class BackgroundImageHandler():
    def __init__(self, img_count, timer):
        self.reset_bkg(img_count, timer)

    def img_accumulator(self, img):
        
        if self._bkg_img is None:
            self._bkg_img = img.astype(np.float32)
            self._avg_counter -= 1

        elif self._avg_counter > 0 and (time.time() - self._last_frame_time) > self._snapshot_timer:
            self._last_frame_time = time.time()
            self._avg_counter -= 1
            self._bkg_img += img.astype(np.float32)
            print(self._avg_counter)
            
        elif self._avg_counter == 0:
            self._bkg_img = (self._bkg_img / self._img_count).astype("uint8")
            self._bkg_state = True

    def reset_bkg(self, img_count, timer):
        self._bkg_state = False
        self._bkg_img = None

        self._img_count = img_count
        self._avg_counter = img_count
        
        self._snapshot_timer = timer
        self._last_frame_time = 0

    def get_bkg_img(self):
        return self._bkg_img

    def get_bkg_state(self):
        return self._bkg_state