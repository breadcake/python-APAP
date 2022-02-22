import numpy as np
import random
from homography import homography_fit, homography_res
import utils
import cv2 as cv

random.seed(0)


class RANSAC:
    def __init__(self, M, thr, visual=True):
        self.psize = 4
        self.M = M
        self.thr = thr
        self.visual = visual

    def __call__(self, img1, img2, src, dst):
        res = self.sampling(src, dst)
        con = sum(res <= self.thr)
        maxinx = np.argmax(con)
        matchesMask = (res[:, maxinx] <= self.thr) + 0
        inliers = np.nonzero(matchesMask)[0]

        if self.visual:
            img3 = utils.cv_draw_matches(img1, img2, src, dst, inliers)
            cv.imshow("ransac's result", img3)
            cv.waitKey(1)

        return src[:, inliers], dst[:, inliers]

    def sampling(self, src, dst):
        n = src.shape[1]
        res = np.zeros((n, self.M))

        for m in range(self.M):
            pinx = random.sample(list(range(n)), self.psize)
            psub1 = src[:, pinx]
            psub2 = dst[:, pinx]

            st = homography_fit(psub1, psub2)

            ds = homography_res(st, src, dst)

            res[:, m] = ds

        return res
