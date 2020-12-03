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

        self.frame_arr = deque([], maxlen=6)
        self.frame_avg = None

    def expand_region(self, region, rad):
        shape = np.shape(self.diff_img)
        x1 = region[0][0]-rad if region[0][0]-rad > 0 else 0
        y1 = region[0][1]-rad if region[0][1]-rad > 0 else 0
        x2 = region[1][0]+rad if region[1][0]+rad < shape[1]-1 else shape[1]-1
        y2 = region[1][1]+rad if region[1][1]+rad < shape[0]-1 else shape[0]-1
        return [(x1, y1), (x2, y2)]

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
        kernel = np.ones((6,6),np.uint8)
        self.binary_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_OPEN, kernel)
        self.binary_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    def scan_for_keypoints(self):
        # Detect contours in the Binary Image
        self.contours, hierarchy = cv2.findContours(self.binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Clear keypoints between frame scans
        self.keypoints = []

        for c in self.contours:
            A = cv2.contourArea(c)
            
            if A < 1400:
                x,y,w,h = cv2.boundingRect(c)
                rect = self.expand_region([[x,y],[x+w,y+h]], 20)
                cv2.rectangle(self.diff_img, rect[0], rect[1],(0,0,255),2)

            elif A < 2500:
                x,y,w,h = cv2.boundingRect(c)
                rect = self.expand_region([[x,y],[x+w,y+h]], 10)
                cv2.rectangle(self.diff_img, rect[0], rect[1],(0,0,255),2)
                
            else:
                x,y,w,h = cv2.boundingRect(c)
                rect = self.expand_region([[x,y],[x+w,y+h]], 10)
                cv2.rectangle(self.diff_img, rect[0], rect[1],(0,0,255),2)

            M = cv2.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(self.diff_img, "Area: " + str(A), (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
ObjClassifier.preprocess_for_scan(diff, 100)
ObjClassifier.scan_for_keypoints()

for c in ObjClassifier.contours:
    x,y,w,h = cv2.boundingRect(c)
    rect = ObjClassifier.expand_region([[x,y],[x+w,y+h]], 20, np.shape(img))
    cv2.rectangle(img, rect[0], rect[1],(0,0,255),2)

cv2.imshow("Centroids", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#numpy.linalg.norm(a-b)
"""