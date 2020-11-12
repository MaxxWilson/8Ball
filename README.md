# EE371Q Project "8-Ball"
### Maxx Wilson, Upayan Mathkuri, Christopher Zum Mallen

## Project Description
This project is intended to enhance the pool playing capabilities through the use of image processing techniques. With a camera positioned above the pool table, we intend to detect and identify different billiards balls and pool cues, noting their location and orientation. This will allow for the prediction of ball trajectory given the angle of a cue to assist a player when lining up shots. The software should also be able to calculate the result of an interaction between two balls, allowing a user to shoot the cue ball at another ball and know where that ball will go. The software could also potentially distinguish between striped and solid balls and identify the ideal shots given a table setup.

## Approach and Methods
### Background Removal
The first task is to identify balls and cues. It becomes much easier to identify objects of interest after removing the background, so we took a picture of the empty pool table and use absdiff() from OpenCV to isolate the balls and cues in the image.

### Identifying Balls and Cues
OpenCV includes an implementation of the Hough Transform in the HoughCircles() function. When tuned properly, this allows us to identify cue balls as circles in the image, returning their X-Y Coordinates and Radius
