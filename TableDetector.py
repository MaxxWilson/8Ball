import numpy as np
import cv2
import time
from ShorthandFunctions import *

img = cv2.imread("low_light2/4.png")
bkg = cv2.imread("Background2.png")
dn_bkg = cv2.fastNlMeansDenoisingColored(bkg,None,10,10,7,21)

print(np.max(img))
print(np.max(FSCS(img)))

cv2.imshow("dn_bkg", FSCS(img))

""" Color Masking """
hsv = cv2.cvtColor(dn_bkg, cv2.COLOR_BGR2HSV)

lower_blue = cv2.cvtColor(np.uint8([[dn_bkg[540, 960]]]), cv2.COLOR_BGR2HSV)
upper_blue = cv2.cvtColor(np.uint8([[dn_bkg[175, 225]]]), cv2.COLOR_BGR2HSV)

lo_sq = np.full((1000, 1000, 3), dn_bkg[540, 960], dtype=np.uint8) / 255.0
hi_sq = np.full((1000, 1000, 3), dn_bkg[175, 225], dtype=np.uint8) / 255.0

cv2.imshow('lo_sq', lo_sq)
cv2.imshow('hi_sq', hi_sq)


# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, np.array([[[70, 0, 20]]]), np.array([[[140, 255, 255]]]))

# Apply closing operation
kernel = np.ones((10,10),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

x,y,w,h = cv2.boundingRect(close)
rect = cv2.cvtColor(close, cv2.COLOR_GRAY2BGR)
rect = cv2.rectangle(rect,(x,y),(x+w,y+h),(0,255,0),2)

rect = cv2.resize(rect, (1536, 864))

cv2.imshow("rectangle?", rect)

# Bitwise-AND mask and original image
#res = cv2.bitwise_and(dn_bkg, dn_bkg, mask=mask)
cv2.imshow('mask',mask)
cv2.imshow('close',close)

#cv2.imwrite("BinaryFeltImage.png", close)
#cv2.imshow('res',res)

impause()


""" Thresholding """
# Convert the img to grayscale 
bkg_gray = cv2.cvtColor(dn_bkg, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian filter to reduce image noise
blur_bkg = cv2.GaussianBlur(bkg_gray,(5,5),0)

cv2.imshow("blurred", blur_bkg)

# Apply Binary thresholding with low threshold to highlight balls
ret, bin_bkg = cv2.threshold(blur_bkg, 15, 255,cv2.THRESH_BINARY)
cv2.imshow("Binary Background", bin_bkg)

# Apply closing operation
kernel = np.ones((10,10),np.uint8)
closing = cv2.morphologyEx(bin_bkg, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Closed Background", closing)

impause()