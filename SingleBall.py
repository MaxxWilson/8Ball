import numpy as np
import time
import matplotlib.pyplot as plt

from ShorthandFunctions import *

# Load two images
img = cv2.imread("low_light2/4.png")
bkg = cv2.imread("Background2.png")

# Desnoise the background image to remove noise in our final difference
dn_bkg = cv2.fastNlMeansDenoisingColored(bkg,None,10,10,7,21)
#cv2.imshow("img", img)

blueball = img[200:300, 650:750]
cv2.imshow("blueball", blueball)

# Convert the img to grayscale 
diff_gray = cv2.cvtColor(blueball, cv2.COLOR_BGR2GRAY)
# Apply a Gaussian filter to reduce image noise
blur = cv2.GaussianBlur(diff_gray,(5,5),0)
cv2.imshow("blur", blur)

th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 6)
# Apply Binary thresholding with low threshold to highlight balls
ret,thresh1 = cv2.threshold(blur, 30, 255,cv2.THRESH_BINARY)

cv2.imshow("Thresholded", thresh1)
cv2.imshow("Adaptive", th2)

cv2.imshow("Edges", cv2.Canny(blur, 0, 10))

# detect circles in the image
circles = cv2.HoughCircles(th2, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=7, minRadius = 20, maxRadius = 25)

print(circles)

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(blueball, (x, y), r, (0, 0, 255), 4)
        cv2.rectangle(blueball, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
    # show the output image
    cv2.imshow("output", blueball)
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()