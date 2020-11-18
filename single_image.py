import numpy as np
import time
import matplotlib.pyplot as plt

from ShorthandFunctions import *

# Load two images
img = cv2.imread("low_light2/4.png")
bkg = cv2.imread("Background2.png")

# Desnoise the background image to remove noise in our final difference
dn_bkg = cv2.fastNlMeansDenoisingColored(bkg,None,10,10,7,21)
cv2.imshow("img", img)

# Calculate absolute difference and display
diff = cv2.absdiff(dn_bkg, img)
cv2.imshow("difference", diff)

# Convert the img to grayscale 
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian filter to reduce image noise
blur = cv2.GaussianBlur(diff_gray,(5,5),0)

# Apply Binary thresholding with low threshold to highlight balls
ret,thresh1 = cv2.threshold(diff_gray, 15, 255,cv2.THRESH_BINARY)

cv2.imshow("Threshold", thresh1)

# Apply an 8x8 morphological Opening operation to remove noise from binary image
kernel = np.ones((8,8),np.uint8)
opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

felt = cv2.imread("BinaryFeltImage.png", cv2.THRESH_BINARY)
cv2.imshow("felt", felt)

print(np.shape(felt))
print(np.shape(opening))

opening = cv2.bitwise_and(opening, felt)

cv2.imshow("Open", opening)

impause()

edges = cv2.Canny(opening, 20, 100)
cv2.imshow("Edges", edges)


# Ball is 48 pixels wide

# detect circles in the image
circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=7, minRadius = 20, maxRadius = 25)

print(circles)

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(diff, (x, y), r, (0, 0, 255), 4)
        cv2.rectangle(diff, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
    # show the output image
    cv2.imshow("output", diff)
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()