import numpy as np
import cv2
import time

# Load two images
img = cv2.imread("test_images/1.png")
bkg = cv2.imread("Background.png")

# Show both images

cv2.imshow("img", img)

# Calculate absolute difference and display
diff = cv2.absdiff(bkg, img)
cv2.imshow("difference", diff)

# Convert the img to grayscale 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(img_gray, 45,255,cv2.THRESH_BINARY)
cv2.imshow("threshold", thresh1)

thresh = 150

edges = cv2.Canny(diff_gray, thresh, thresh+50, 9)
cv2.imshow("Edges", edges)


"""
# This returns an array of r and theta values 
lines = cv2.HoughLines(edges,1,np.pi/180, 50, 100)

print(lines)

x1 = 0
x2 = 0
y1 = 0
y2 = 0

# The below for loop runs till r and theta values  
# are in the range of the 2d array
for i in range(len(lines)):
    for r,theta in lines[i]: 
        
        # Stores the value of cos(theta) in a 
        a = np.cos(theta) 
    
        # Stores the value of sin(theta) in b 
        b = np.sin(theta) 
        
        # x0 stores the value rcos(theta) 
        x0 = a*r 
        
        # y0 stores the value rsin(theta) 
        y0 = b*r 
        
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
        x1 += int(x0 + 1000*(-b)) 
        
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
        y1 += int(y0 + 1000*(a)) 
    
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
        x2 += int(x0 - 1000*(-b)) 
        
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
        y2 += int(y0 - 1000*(a)) 
        
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
        # (0,0,255) denotes the colour of the line to be  
        #drawn. In this case, it is red.  
        
cv2.line(img,(x1//5,y1//5), (x2//5,y2//5), (0,0,255),1) 

cv2.imshow("lines?", img)

"""
"""
# Ball is 48 pixels wide

# detect circles in the image
circles = cv2.HoughCircles(diff_gray, cv2.HOUGH_GRADIENT, 1, 40, param1=200, param2=15, minRadius = 20, maxRadius = 45)

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
"""
"""
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(img_gray)
print(keypoints)
im_with_keypoints = cv2.drawKeypoints(img_gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)

"""
cv2.waitKey(0)
cv2.destroyAllWindows()