import numpy as np
import cv2

#given a cue ball point and a vector
cue = [305,570]
vect = [0.792626,-0.6097127]   #vect = [vx,vy,cue_x,cue_y]
slope = vect[1]/vect[0]
r = 20 #Ball radius. Not sure what units but doesn't matter all that much

#and given a variety of other balls
balls = [[600,330]]

#Find whether there is a ball that the cue will hit
#1) draw lines parallel to cue trajectory +- 2R from center of cue
#Line1 = [vect[0],vect[1],cue[0]+vect[0]*2*r,cue[1]+vect[1]*2*r]
#Line2 = [vect[0],vect[1],cue[0]-vect[0]*2*r,cue[1]-vect[1]*2*r]

#~~~~Image to draw circle and lines on
img = cv2.imread('Fin1.png')
cv2.circle(img,(cue[0],cue[1]),r,(0,255,255),3)
 

#2) Find closest ball between the two lines
for i in range(0,len(balls)):
    #find the vector from the cue ball to the other ball
    x_diff = balls[i][0]-cue[0] #find the difference between the ball and cue
    y_diff = balls[i][1]-cue[1]
    ball_dist = np.sqrt(x_diff**2+y_diff**2)  #find the vector magnitude to the ball
    vect_temp = [x_diff/ball_dist, y_diff/ball_dist] #normalized vector to ball
    slope_temp = vect_temp[1]/vect_temp[0] #slope of normalized vector
    theta = np.arctan((np.abs(slope-slope_temp))/(1+slope*slope_temp)) #angle between two vectors
    norm_dist = np.sin(theta)*ball_dist #distance between the ball and the original vector
    if norm_dist < 2*r:
        phi = np.arcsin(norm_dist/(2*r)) #angle of impact (from original line)
        length = np.cos(theta)*ball_dist - np.cos(phi)*2*r #distance along original line to impact point
        #find a normalized vector of the line of impact
        impact = [cue[0]+vect[0]*length,cue[1]+vect[1]*length]     
        impvect = [(balls[i][0]-impact[0])/(2*r),(balls[i][1]-impact[1])/(2*r)] #Line of impact normalized vector
        line = [impvect[0],impvect[1],balls[i][0],balls[i][1]] #vect = [vx,vy,cue_x,cue_y]
        
        #~~~~
        cv2.line(img,(line[2],line[3]),(int(line[2]+line[0]*1000),int(line[3]+line[1]*1000)),(255,0,255),2)
        cv2.circle(img,(int(impact[0]),int(impact[1])),r,(0,255,255),3)
        cv2.line(img,(cue[0],cue[1]),(int(impact[0]),int(impact[1])),(0,255,255),2)
        
    #~~~~
    cv2.circle(img,(balls[i][0],balls[i][1]),r,(0,255,0),3)

cv2.imwrite('FinalExample.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()