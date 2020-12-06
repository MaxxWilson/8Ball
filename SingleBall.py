"""
Scratch file for testing various operations on single ball regions.
"""


import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

# Load two images
large2 = cv2.imread("test_regions/Medium5.png")

white = [255, 255, 255]  # RGB
diff = 60
boundaries = [([white[2]-diff, white[1]-diff, white[0]-diff],[white[2], white[1], white[0]])]

def colorPicker (img, boundaries):
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        
        print("white pix count: ", cv2.countNonZero(mask))
        #ratio_white = cv2.countNonZero(mask)/(img.size/3)
        #print('white pixel percentage:', np.round(ratio_white*100, 2))

        cv2.imshow("images", np.hstack([img, output]))
        #cv2.waitKey(0)

colorPicker(large2, boundaries)

"""
# Convert the img to grayscale 
diff_gray = cv2.cvtColor(large2, cv2.COLOR_BGR2GRAY)
# Apply a Gaussian filter to reduce image noise
#blur = cv2.GaussianBlur(diff_gray,(5,5),0)
blur = diff_gray
cv2.imshow("blur", blur)

th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6)
# Apply Binary thresholding with low threshold to highlight balls
ret,thresh1 = cv2.threshold(blur, 20, 255,cv2.THRESH_BINARY)

cv2.imshow("Thresholded", thresh1)
cv2.imshow("Adaptive", th2)

cv2.imshow("Edges", cv2.Canny(thresh1, 0, 45))

# detect circles in the image
circles = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=7, minRadius = 20, maxRadius = 25)[0, :]

print(circles)

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(large2, (x, y), r, (0, 0, 255), 4)
        cv2.rectangle(large2, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
    # show the output image
    cv2.imshow("output", large2)
    cv2.waitKey(0)
"""
cv2.waitKey(0)
cv2.destroyAllWindows()