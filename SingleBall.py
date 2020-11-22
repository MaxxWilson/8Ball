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

# Apply Binary thresholding with low threshold to highlight balls
ret,thresh1 = cv2.threshold(blur, 30, 255,cv2.THRESH_BINARY)

cv2.imshow("Thresholded", thresh1)

cv2.imshow("Edges", cv2.Canny(blur, 0, 10))

cv2.waitKey(0)
cv2.destroyAllWindows()