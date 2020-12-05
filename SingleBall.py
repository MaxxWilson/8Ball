import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
# Load two images
large2 = cv2.imread("test_regions/Multi Ball2_diff.png")

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

cv2.waitKey(0)
cv2.destroyAllWindows()