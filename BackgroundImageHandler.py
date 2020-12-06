"""
Defines a class for handling the background of the pool table. Called when we want to recapture a background image for
background removal, averaging the video feed over a time period to reduce noise level
s"""

import time
import numpy as np
import cv2


class BackgroundImageHandler():
    def __init__(self, img_count = 30, timer = 0.01):
        self.reset_bkg(img_count, timer)

        self._bkg_img_gray = None
        self._bkg_img_thresh = None

        self.bounding_rect = []
        
        self.table_thickness = 100

        self.debug_toggle = False

    def img_accumulator(self, img):
        # On first pass, grab the background image
        if self._avg_counter == self._img_count:
            self._bkg_img = img.astype(np.float32, copy=False)
            self._avg_counter -= 1

        # After first image is captured, add new image at set interval as counter decreases
        elif self._avg_counter > 0 and (time.time() - self._last_frame_time) > self._snapshot_timer:
            self._last_frame_time = time.time()
            self._avg_counter -= 1
            self._bkg_img += img.astype(np.float32, copy=False)
            
        # When count runs out, average all the background images, break the loop by setting bkg_state to true, and calculate table border
        elif self._avg_counter == 0:
            self._bkg_img = (self._bkg_img / self._img_count).astype("uint8", copy=False)
            self._bkg_state = True
            self.calculate_table_border(8)

    def calculate_table_border(self, threshold):
        # Convert to background image to grayscale and median blur with 5x5 kernel
        self._bkg_img_gray = cv2.medianBlur(cv2.cvtColor(self._bkg_img, cv2.COLOR_BGR2GRAY), 5)

        # Apply Inverse Binary threshold to isolate table from surroundings.
        _, self._bkg_img_thresh = cv2.threshold(self._bkg_img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Run a close-open operation with two iterations to clean up noise
        kernel = np.ones((10,10),np.uint8)
        self.binary_img = cv2.morphologyEx(self._bkg_img_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        self.binary_img = cv2.morphologyEx(self._bkg_img_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Fit a rectangle to the filtered table border
        x, y, w, h = cv2.boundingRect(self._bkg_img_thresh)

        # Apply constant offset from the table border to roughly isolate the inner walls of the felt
        self.bounding_rect = [[x+self.table_thickness, y+self.table_thickness], [x+w-self.table_thickness, y+h-self.table_thickness]]

    def reset_bkg(self, img_count, timer):
        # Reset class state to capture a new background image
        self._bkg_state = False
        self._bkg_img = None

        self._img_count = img_count
        self._avg_counter = img_count
        
        self._snapshot_timer = timer
        self._last_frame_time = 0

    def toggle_table_debug(self):
        self.debug_toggle = not self.debug_toggle

    def save_background(self, file="Background.png"):
        cv2.imwrite(file, self._bkg_img)

    def load_background(self, file="Background.png"):
        self._bkg_state = True
        self._bkg_img = cv2.imread(file)

    def get_bkg_img(self):
        return self._bkg_img

    def get_bkg_state(self):
        return self._bkg_state

    def get_table_border(self):
        return self.bounding_rect

"""
# Test for calculate_table_border()

BkgHandler = BackgroundImageHandler()
BkgHandler.load_background("BackgroundAvg.png")
cv2.imshow("Background", BkgHandler.get_bkg_img())
cv2.waitKey(0)
cv2.destroyAllWindows()

rect = BkgHandler.calculate_table_border(10)

cv2.imshow("Thresholded Image", BkgHandler._bkg_img_thresh)

#rect = cv2.rectangle(BkgHandler.get_bkg_img().copy(),(rect[0][0],rect[0][1]),(rect[1][0],rect[1][1]),(0,255,0),2)
cv2.imshow("Bounds", BkgHandler.get_bkg_img()[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]])

cv2.waitKey(0)
cv2.destroyAllWindows()
"""