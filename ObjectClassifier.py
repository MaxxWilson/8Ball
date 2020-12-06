"""Defines a class to search for possible balls and cues, identify their geometry in the image, and track them over time"""

from collections import deque
import numpy as np
import cv2

class ObjectClassifier():
    def __init__(self):
        self.keypoints = []
        self.diff_gray = None
        self.binary_img = None
        self.contours = None

        self.circles = []

        self.frame_arr = deque([], maxlen=4)
        self.frame_avg = None

    def preprocess_for_scan(self, img, diff_img, ball_threshold):
        self.img = img
        self.diff_img = diff_img

        # Convert the img to grayscale 
        self.diff_gray = cv2.cvtColor(self.diff_img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian filter to reduce image noise
        self.diff_gray = cv2.GaussianBlur(self.diff_gray,(11,11),0)
        
        #### Frame Averaging for Difference Image Feed ####
        if len(self.frame_arr) == 0:
            self.frame_arr.append(self.diff_gray.astype(np.float32, copy=False))
            self.frame_avg = self.diff_gray

        elif(np.shape(self.frame_arr[0]) != np.shape(self.diff_gray)):
            self.frame_arr.clear()
            self.frame_arr.append(self.diff_gray.astype(np.float32, copy=False))
            self.frame_avg = self.diff_gray

        else:
            self.frame_arr.append(self.diff_gray.astype(np.float32, copy=False))
            self.frame_avg = (np.sum(self.frame_arr, 0)/len(self.frame_arr)).astype("uint8", copy=False)

        # Apply Binary thresholding with low threshold to highlight balls
        _, self.binary_img = cv2.threshold(self.frame_avg, ball_threshold, 255,cv2.THRESH_BINARY)

        # Apply morphological Opening operation to remove noise from binary image
        kernel = np.ones((8,8),np.uint8)
        self.binary_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_OPEN, kernel)
        self.binary_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    def scan_for_key_regions(self):
        # Detect contours in the Binary Image
        self.contours, hierarchy = cv2.findContours(self.binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Clear search regions between frame scans
        self.search_regions = {"Ball Segment":[], "Single Ball":[], "Multi Ball":[], "Cue":[]}

        for c in self.contours:
            # For each contour, calculate area and bounding box
            A = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)

            # Filter by area and percentage of area covered
            if A < 1400:
                radius_pad = 20
                region_type = "Ball Segment"
            elif A < 2500:
                radius_pad = 10
                region_type = "Single Ball"
            else:
                radius_pad = 10
                if A/(w*h) > 0.5:
                    region_type = "Multi Ball"
                else:
                    region_type = "Cue"
            
            # Apply region expansion with pad radius and append to search regions dictionary
            self.search_regions[region_type].append(self.expand_region([[x,y],[x+w,y+h]], radius_pad))

    def expand_region(self, region, rad):
        shape = np.shape(self.diff_img)
        x1, y1 = max(region[0][0]-rad, 0), max(region[0][1]-rad, 0)
        x2, y2 = min(region[1][0]+rad, shape[1]-1), min(region[1][1]+rad, shape[0]-1)
        return np.array([(x1, y1), (x2, y2)])

    def identify_balls(self):
        self.circles = []

        for rect in self.search_regions.get("Ball Segment"):
            img_rgn = self.frame_avg[rect[0, 1]:rect[1, 1], rect[0, 0]:rect[1, 0]]
            self.region_detect_ball_segment(img_rgn, rect[0])

        for rect in self.search_regions.get("Single Ball"):
            img_rgn = self.frame_avg[rect[0, 1]:rect[1, 1], rect[0, 0]:rect[1, 0]]
            self.region_detect_single_ball(img_rgn, rect[0])

        for rect in self.search_regions.get("Multi Ball"):
            img_rgn = self.frame_avg[rect[0, 1]:rect[1, 1], rect[0, 0]:rect[1, 0]]
            self.region_detect_multi_ball(img_rgn, rect[0])

        for rect in self.search_regions.get("Cue"):
            img_rgn = self.frame_avg[rect[0, 1]:rect[1, 1], rect[0, 0]:rect[1, 0]]
            self.region_detect_cue(img_rgn, rect[0])
    
    def region_detect_ball_segment(self, im_rgn, lead_point):
        # Ball detector for regions with small contours
        _, thresh = cv2.threshold(im_rgn, 10, 255,cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=7, minRadius = 15, maxRadius = 25)
        self.circles.append(self.transform_circles_to_full_img(lead_point, circles))

    def region_detect_single_ball(self, im_rgn, lead_point):
        # Ball detector for regions with contours the size of roughly a single ball
        _, thresh = cv2.threshold(im_rgn, 15, 255,cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=7, minRadius = 15, maxRadius = 25)
        self.circles.append(self.transform_circles_to_full_img(lead_point, circles))

    def region_detect_multi_ball(self, im_rgn, lead_point):
        # Ball detector for regions pre-classified as containing multiple balls
        _, thresh = cv2.threshold(im_rgn, 15, 255,cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=7, minRadius = 15, maxRadius = 25)
        self.circles.append(self.transform_circles_to_full_img(lead_point, circles))

    def region_detect_cue(self, im_rgn, lead_point):
        # Cue detector for regions of pre-classified as cues

        # Apply high threshold to cue region
        _, thresh = cv2.threshold(im_rgn, 81, 255, cv2.THRESH_BINARY)
        cv2.imshow("CUE", thresh)

        # Identify contours in new thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # break from function if no contours are detected
        if len(contours) == 0:
            return

        # Calculate the contour areas to identify largest controur
        area = []
        for c in contours:
            area.append(cv2.contourArea(c))

        # Fit line to the contour of largest area, returning vector defining angle and a point on the line
        [vx,vy,x,y] = cv2.fitLine(contours[np.where(np.array(area)==np.max(area))[0][0]], cv2.DIST_L2,0,0.01,0.01)

        # Transform coordinate system back to full scale image
        x2 = x + lead_point[0]
        y2 = y + lead_point[1]

        cols = np.shape(self.diff_img)[1]
        lefty = int((-x2*vy/vx) + y2)
        righty = int(((cols-x2)*vy/vx)+y2)
        cv2.line(self.diff_img,(cols-1,righty),(0,lefty),(0,255,0),2)

    def transform_circles_to_full_img(self, rectangle, circles):
        if circles is not None:
            circles = (circles[0, :]).astype("int")
            for i in range(len(circles)):
                circles[i][0] += rectangle[0]
                circles[i][1] += rectangle[1]
        return circles

    def draw_search_regions(self):
        for region in self.search_regions.keys():
            for rect in self.search_regions.get(region):
                cv2.rectangle(self.diff_img, tuple(rect[0]), tuple(rect[1]),(0,0,255),2)
        return self.diff_img
    
    def draw_circles(self):
        for circles in self.circles:
            if circles is not None:
                for (x, y, r) in circles:
                    cv2.circle(self.diff_img, (x, y), r, (0, 0, 255), 4)
                    cv2.rectangle(self.diff_img, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
        return self.diff_img
    
    def save_regions(self):
        PATH = "/home/maxx/DIP/8Ball/test_regions/"
        for region in self.search_regions.keys():
            picture_count = 1
            for rect in self.search_regions.get(region):
                cv2.imwrite(PATH + region + str(picture_count) + "_diff.png", self.diff_img[rect[0, 1]:rect[1, 1], rect[0, 0]:rect[1, 0]])
                picture_count += 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()



"""
# Load two images
img = cv2.imread("high_light/2.png")
bkg = cv2.imread("Background2.png")

# Desnoise the background image to remove noise in our final difference
dn_bkg = cv2.fastNlMeansDenoisingColored(bkg,None,10,10,7,21)
#cv2.imshow("img", img)

# Calculate absolute difference and display
diff = cv2.absdiff(dn_bkg, img)

ObjClassifier = ObjectClassifier()
ObjClassifier.preprocess_for_scan(diff, 100)
#ObjClassifier.scan_for_key_regions()
contours, hierarchy = cv2.findContours(ObjClassifier.binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

centroids = []

count = 1
for i in range(len(contours)):
    A = cv2.contourArea(contours[i])

    x,y,w,h = cv2.boundingRect(contours[i])

    if A < 1400:
        rect = ObjClassifier.expand_region([[x,y],[x+w,y+h]], 20)
    elif A < 2500:
        rect = ObjClassifier.expand_region([[x,y],[x+w,y+h]], 10)
    else:
        rect = ObjClassifier.expand_region([[x,y],[x+w,y+h]], 10)

    cv2.rectangle(diff, rect[0], rect[1],(0,0,255),2)

    count += 1

cv2.imshow("Centroids", ObjClassifier.diff_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""