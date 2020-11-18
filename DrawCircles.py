


def DrawCircles(binary, img):
    # detect circles in the image
    circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=7, minRadius = 20, maxRadius = 25)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img, (x, y), r, (0, 0, 255), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
        # show the output image
        cv2.imshow("output", img)
        cv2.waitKey(0)