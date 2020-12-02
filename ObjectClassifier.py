"""Defines a class to search for possible balls and cues, identify their geometry in the image, and track them over time"""


import numpy as np
import cv2

class ObjectClassifier():
    def __init__(self):
        self.keypoints = []
        self.diff_gray = None
        self.binary_img = None
        self.contours = None
        

    def scan_for_keypoints(self, diff_img):
        # Convert the img to grayscale 
        self.diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian filter to reduce image noise
        self.diff_gray = cv2.GaussianBlur(self.diff_gray,(9,9),0)
        
        # Apply Binary thresholding with low threshold to highlight balls
        _, self.binary_img = cv2.threshold(self.diff_gray, 15, 255,cv2.THRESH_BINARY)
        cv2.imshow("Binary Image", self.binary_img)

        # Apply morphological Opening operation to remove noise from binary image
        kernel = np.ones((10,10),np.uint8)
        self.binary_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_OPEN, kernel)
        self.binary_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_CLOSE, kernel)

        self.contours, hierarchy = cv2.findContours(self.binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.keypoints = []

        for c in self.contours:
            M = cv2.moments(c)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.keypoints.append([cX, cY, ])

    def classify_striped_solid(self):
        pass

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
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Centroids", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


numpy.linalg.norm(a-b)