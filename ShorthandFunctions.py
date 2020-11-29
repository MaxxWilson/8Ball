""" Shorthand Functions """

# Use cv2.normalize to apply FSCS and scale large number matrices back to image format
def FSCS(image):
    return cv2.normalize(image.copy(), None, 0, 255, cv2.NORM_MINMAX, dtype=0)

def impause():
    cv2.waitKey(0)
    cv2.destroyAllWindows()