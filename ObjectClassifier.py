"""Defines a class to search for possible balls and cues, identify their geometry in the image, and track them over time"""

from collections import deque
import numpy as np
import cv2

class ObjectClassifier():
    def __init__(self):
        self.keypoints = []
        self.diff_gray = None
        self.binary_img = None
        self.contours = None

        self.frame_arr = deque([], maxlen=4)
        self.frame_avg = None
        

    def preprocess_for_scan(self, diff_img, ball_threshold):
        self.diff_img = diff_img

        cv2.imshow("Diff Img", self.diff_img)

        # Convert the img to grayscale 
        self.diff_gray = cv2.cvtColor(self.diff_img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian filter to reduce image noise
        self.diff_gray = cv2.GaussianBlur(self.diff_gray,(11,11),0)
        
        if len(self.frame_arr) == 0:
            self.frame_arr.append(self.diff_gray.astype(np.float32, copy=False))
            self.frame_avg = self.diff_gray
        elif(np.shape(self.frame_arr[0]) != np.shape(self.diff_gray)):
            self.frame_arr.clear()
            self.frame_arr.append(self.diff_gray.astype(np.float32, copy=False))
            self.frame_avg = self.diff_gray
        else:
            self.frame_arr.append(self.diff_gray.astype(np.float32, copy=False))
            self.frame_avg = (np.sum(self.frame_arr, 0)/len(self.frame_arr)).astype("uint8", copy=False)

        cv2.imshow("Frame Avg", self.frame_avg)

        # Apply Binary thresholding with low threshold to highlight balls
        _, self.binary_img = cv2.threshold(self.frame_avg, ball_threshold, 255,cv2.THRESH_BINARY)
        cv2.imshow("Binary Image", self.binary_img)

        # Apply morphological Opening operation to remove noise from binary image
        kernel = np.ones((7,7),np.uint8)
        self.binary_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_OPEN, kernel)
        self.binary_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_CLOSE, kernel)

    def scan_for_keypoints(self):
        # Detect contours in the Binary Image
        self.contours, hierarchy = cv2.findContours(self.binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Clear keypoints between frame scans
        self.keypoints = []

        for c in self.contours:
            A = cv2.contourArea(c)

            if A < 50:
                # If the area is below threshold, it is noise
                continue
            elif A < 1000:
                cv2.putText(img, "Dark Ball", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif A < 2500:
                cv2.putText(img, "Single Ball", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(img, "Cue or Multi-Ball", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            #M = cv2.moments(c)
            #self.keypoints.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(self.diff_img,(x,y),(x+w,y+h),(0,255,0),2)

            #rect = cv2.minAreaRect(c)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #cv2.drawContours(self.diff_img,[box],0,(0,0,255),2)
            
        return self.diff_img

    def classify_striped_solid(self):
        pass



"""
# Load two images
img = cv2.imread("high_light/2.png")
bkg = cv2.imread("Background2.png")

# Desnoise the background image to remove noise in our final difference
dn_bkg = cv2.fastNlMeansDenoisingColored(bkg,None,10,10,7,21)
#cv2.imshow("img", img)

# Calculate absolute difference and display
diff = cv2.absdiff(dn_bkg, img)

ObjClassifier = ObjectClassifier()
ObjClassifier.scan_for_keypoints(diff)

for c in ObjClassifier.contours:
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)

cv2.imshow("Centroids", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#numpy.linalg.norm(a-b)
"""