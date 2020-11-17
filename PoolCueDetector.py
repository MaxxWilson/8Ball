
import numpy as np
import cv2
import time

# Load two images
img = cv2.imread("low_light2/4.png")
bkg = cv2.imread("Background2.png")
dn_bkg = cv2.fastNlMeansDenoisingColored(bkg,None,10,10,7,21)

# Calculate absolute difference and display
diff = cv2.absdiff(dn_bkg, img)
cv2.imshow("difference", diff)

# Convert the img to grayscale 
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

#
#blurred = cv2.medianBlur(diff_gray, 7)

# Apply Binary thresholding to find balls
blur = cv2.GaussianBlur(diff_gray,(5,5),0)
ret,thresh1 = cv2.threshold(blur, 17,255,cv2.THRESH_BINARY)
cv2.imshow("Threshold", thresh1)
#cv2.medianBlur(thresh1, 5)

edges = cv2.Canny(thresh1, 20, 100)
cv2.imshow("Edges", edges)

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