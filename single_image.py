import numpy as np
import time
import matplotlib.pyplot as plt
import cv2


from ShorthandFunctions import *

# Load two images
img = cv2.imread("low_light2/10.png")
bkg = cv2.imread("Background2.png")

# Desnoise the background image to remove noise in our final difference
dn_bkg = cv2.fastNlMeansDenoisingColored(bkg,None,10,10,7,21)
#cv2.imshow("img", img)

# Calculate absolute difference and display
diff = cv2.absdiff(dn_bkg, img)
cv2.imshow("difference", diff)

# Convert the img to grayscale 
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian filter to reduce image noise
blur = cv2.GaussianBlur(diff_gray,(9,9),0)

# Apply Binary thresholding with low threshold to highlight balls
ret,thresh1 = cv2.threshold(diff_gray, 50, 255,cv2.THRESH_BINARY)

cv2.imshow("Threshold", thresh1)

cv2.waitKey(0)
cv2.destroyAllWindows()


felt = cv2.imread("BinaryFeltImage.png", cv2.THRESH_BINARY)
cv2.imshow("felt", felt)

ball_region = cv2.bitwise_and(thresh1, felt)

#cv2.imshow("Ball Region", ball_region)

# Apply an 8x8 morphological Opening operation to remove noise from binary image
kernel = np.ones((10,8),np.uint8)
opening = cv2.morphologyEx(ball_region, cv2.MORPH_OPEN, kernel)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Close", close)

contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_img = cv2.drawContours(img.copy(), contours, -1, (0,255,0), 3)

count = 1
for c in contours:
    print(count)
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    A = cv2.contourArea(c)
    print("Area:", A)
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    print()
    if A < 1000:
        cv2.putText(img, "Dark Ball", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif A < 2500:
        cv2.putText(img, "Single Ball", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Cue or Multi-Ball", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    count += 1

cv2.imshow("Contours", contour_img)
cv2.imshow("Centroids", img)

impause()

#edges = cv2.Canny(opening, 20, 100)
#cv2.imshow("Edges", edges)


# Ball is 48 pixels wide

#cv2.waitKey(0)
#cv2.destroyAllWindows()
