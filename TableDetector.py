"""
Scratch file for working on table detection.
"""


import numpy as np
import cv2
import time

img = cv2.imread("BackgroundAvg.png")

cv2.imshow("dn_bkg", FSCS(img))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Threshold", thresh)

x,y,w,h = cv2.boundingRect(thresh)
rect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("rectangle?", rect)


cv2.waitKey(0)
cv2.destroyAllWindows()

"""
"""" Color Masking """"
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = cv2.cvtColor(np.uint8([[img[540, 960]]]), cv2.COLOR_BGR2HSV)
upper_blue = cv2.cvtColor(np.uint8([[img[175, 225]]]), cv2.COLOR_BGR2HSV)

lo_sq = np.full((1000, 1000, 3), img[540, 960], dtype=np.uint8) / 255.0
hi_sq = np.full((1000, 1000, 3), img[175, 225], dtype=np.uint8) / 255.0

cv2.imshow('lo_sq', lo_sq)
cv2.imshow('hi_sq', hi_sq)


# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, np.array([[[70, 0, 20]]]), np.array([[[140, 255, 255]]]))

# Apply closing operation
kernel = np.ones((10,10),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

x,y,w,h = cv2.boundingRect(close)
rect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

rect = cv2.resize(rect, (1536, 864))

cv2.imshow("rectangle?", rect)

# Bitwise-AND mask and original image
#res = cv2.bitwise_and(dn_bkg, dn_bkg, mask=mask)
cv2.imshow('mask',mask)
cv2.imshow('close',close)

#cv2.imwrite("BinaryFeltImage.png", close)
#cv2.imshow('res',res)

impause()
"""