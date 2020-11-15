import numpy as np
import cv2
import time
imgs = []
start = time.time()

for i in range(5):
    imgs.append(cv2.imread("test_images/" + str(i+1) + ".png"))
    start = time.time()
    imgs[i] = cv2.Canny(imgs[i], 100, 200)
    print(time.time()-start)
    cv2.imshow(str(i), imgs[i])

cv2.waitKey(0)
cv2.destroyAllWindows()