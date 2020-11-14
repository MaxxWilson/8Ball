import numpy as np
import cv2
import time

# Load two images
img1 = cv2.imread("test_images/3.png")
img2 = cv2.imread("test_images/4.png")

img1 = cv2.erode(img1, np.ones((5, 5), np.uint8))

img_edges = cv2.Canny(img1, 100, 200, 9)

cv2.imshow("edges", img_edges)

# Show both images

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)

# Calculate absolute difference and display
diff = cv2.absdiff(img1, img2)

cv2.imshow("difference", diff)
"""
# Convert to Grayscale

img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Show grayscale image and difference

cv2.imshow("img_gray", img_gray)
cv2.imshow("diff gray", diff_gray)

# Ball is 48 pixels wide

# detect circles in the image
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.5, 45, minRadius = 20, maxRadius = 35)

print(circles)
circles = np.array([[(1080/2, 1200/2, 100)]])

# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		output = img_gray.copy()
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	# show the output image
	cv2.imshow("output", output)
	cv2.waitKey(0)
"""
cv2.waitKey(0)
cv2.destroyAllWindows()