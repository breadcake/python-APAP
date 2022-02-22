import cv2 as cv


class matchers():
    def __init__(self):
        self.sift = cv.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

    def match(self, des1, des2):
        matches = self.flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return good

    def getFeatures(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des