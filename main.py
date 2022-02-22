import cv2 as cv
import numpy as np
import math
import time
import matchers
from ransac import RANSAC
from homography import homography_fit, get_hom_final_size
from apap import APAP_stitching, get_mdlt_final_size
from imagewarping import imagewarping
from imageblending import imageblending
import config

def resize_image(img):
    ori_image_size = img.shape[0]*img.shape[1]
    if ori_image_size > 1000*600:
        scale = math.sqrt(1000*600 / ori_image_size)
        img = cv.resize(img, None, fx=scale, fy=scale)
    return img

## Read images
# img1 = cv.imread('images/DHW-temple/4.JPG')
# img2 = cv.imread('images/DHW-temple/5.JPG')
img1 = cv.imread('images/APAP-railtracks/P1010517.JPG')
img2 = cv.imread('images/APAP-railtracks/P1010520.JPG')
img1 = resize_image(img1)    # resize to avoid out of memory
img2 = resize_image(img2)

## SIFT keypoint detection and matching
t0 = time.time()
matcher_obj = matchers.matchers()
kp1, ds1 = matcher_obj.getFeatures(img1)
kp2, ds2 = matcher_obj.getFeatures(img2)
matches = matcher_obj.match(ds1, ds2)
print("Number of matches: ", len(matches))
print('Keypoint detection and matching done in {:.6f}s'.format(time.time() - t0))

src_orig = np.float32([kp1[m.queryIdx].pt for m in matches])
dst_orig = np.float32([kp2[m.trainIdx].pt for m in matches])
src_orig = np.vstack((src_orig.T, np.ones((1, len(matches)))))
dst_orig = np.vstack((dst_orig.T, np.ones((1, len(matches)))))

##################
# Outlier removal.
##################
t1 = time.time()
ransac = RANSAC(config.M, config.thr)
src_fine, dst_fine = ransac(img1, img2, src_orig, dst_orig)
print("Number of inliers: ", src_fine.shape[1])
print('RANSAC sampling done in {:.6f}s'.format(time.time() - t1))

########################
# Global homography (H).
########################
print("\nDLT (projective transform) on inliers")
# Refine homography using DLT on inliers.
Hg = homography_fit(src_fine, dst_fine)

#############################################
# Image stitching with global homography (H).
#############################################
# Obtaining size of canvas (using global Homography)
min_x, max_x, min_y, max_y = get_hom_final_size(img1, img2, Hg)

# warping images
t0 = time.time()
warped_img1, warped_img2, warped_mask1, warped_mask2 = imagewarping(img1, img2, Hg, min_x, max_x, min_y, max_y)
print("> Warping images by global homography done in {:.6f}s".format(time.time() - t0))

# blending
t0 = time.time()
linear_hom = imageblending(warped_img1, warped_img2, warped_mask1, warped_mask2)
print("  Homography linear image blending done in {:.6f}s".format(time.time() - t0))
cv.imshow("linear_hom", linear_hom)
cv.waitKey(1)

##########################
# Moving DLT (projective).
##########################
print('\nAs-Projective-As-Possible Moving DLT on inliers')
start = time.time()
# Generating mesh for MDLT
X, Y = np.meshgrid(np.linspace(0, img2.shape[1]-1, config.C1), np.linspace(0, img2.shape[0]-1, config.C2))
# Mesh (cells) vertices' coordinates
Mv = np.array([X.ravel(), Y.ravel()]).T
# Perform Moving DLT
apap = APAP_stitching(config.gamma, config.sigma)
Hmdlt = apap(dst_fine, src_fine, Mv)
print('Moving DLT main loop done in {:.4f}s'.format(time.time() - start))

##################################
# Image stitching with Moving DLT.
##################################
min_x, max_x, min_y, max_y = get_mdlt_final_size(img1, img2, Hmdlt, config.C1, config.C2)
t0 = time.time()
warped_img1, warped_img2, warped_mask1, warped_mask2 = imagewarping(img1, img2, Hmdlt, min_x, max_x, min_y, max_y,
                                                                    config.C1, config.C2)
print("> Warping images with Moving DLT done in {:.6f}s".format(time.time() - t0))

t0 = time.time()
linear_mdlt = imageblending(warped_img1, warped_img2, warped_mask1, warped_mask2)
print("  Moving DLT linear image blending done in {:.6f}s".format(time.time() - t0))
cv.imshow("linear_mdlt", linear_mdlt)
cv.waitKey(0)
cv.destroyAllWindows()
