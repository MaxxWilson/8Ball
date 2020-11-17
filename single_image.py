import numpy as np
import cv2
import time

# Load two images
img = cv2.imread("low_light2/4.png")
bkg = cv2.imread("Background2.png")
dn_bkg = cv2.fastNlMeansDenoisingColored(bkg,None,10,10,7,21)

cv2.imshow("img", img)

# Calculate absolute difference and display
diff = cv2.absdiff(dn_bkg, img)
cv2.imshow("difference", diff)

# Convert the img to grayscale 
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

#
#blurred = cv2.medianBlur(diff_gray, 7)

# Apply Binary thresholding to find balls
blur = cv2.GaussianBlur(diff_gray,(5,5),0)
ret,thresh1 = cv2.threshold(blur, 15,255,cv2.THRESH_BINARY)
cv2.imshow("Threshold", thresh1)
#cv2.medianBlur(thresh1, 5)

edges = cv2.Canny(thresh1, 20, 100)
cv2.imshow("Edges", edges)


# Ball is 48 pixels wide

# detect circles in the image
circles = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=7, minRadius = 20, maxRadius = 25)

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