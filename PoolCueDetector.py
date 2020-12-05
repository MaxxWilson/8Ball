
import numpy as np
import cv2
import time

img = cv2.imread("test_regions/Cue1_diff.png")

img = img[2:np.shape(img)[0]-2, 2:np.shape(img)[1]-2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 40, 255,cv2.THRESH_BINARY)

kernel = np.ones((3,3),np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for i in contours:
    print(cv2.contourArea(i))

cnt = contours[0]

#rect = cv2.minAreaRect(cnt)
#box = cv2.boxPoints(rect)
#box = np.int0(box)
#cv2.drawContours(img,[box],0,(0,0,255),2)

cv2.imshow("img", img)
cv2.imshow("Thresh", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
print(vx, vy, x, y)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

#ellipse = cv2.fitEllipse(cnt)
#cv2.ellipse(img,ellipse,(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()