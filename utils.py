import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def appendimages(im1, im2):
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.vstack((im1, np.zeros((rows2-rows1,im1.shape[1],3),np.uint8)))
    elif rows1 > rows2:
        im2 = np.vstack((im2, np.zeros((rows1-rows2,im2.shape[1],3),np.uint8)))
    return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, inliers):
    plt.figure()
    im3 = appendimages(im1, im2)
    plt.imshow(im3[:,:,::-1])
    out1, out2 = np.delete(locs1, inliers, axis=1), np.delete(locs2, inliers, axis=1)
    plt.plot(out1[0,:], out1[1,:], 'ro', markerfacecolor='none', linewidth=2)
    plt.plot(out2[0,:]+im1.shape[1], out2[1,:], 'ro', markerfacecolor='none', linewidth=2)
    plt.plot(locs1[0,inliers], locs1[1,inliers], 'o', c='#00FF00', markerfacecolor='none', linewidth=2)
    plt.plot(locs2[0,inliers]+im1.shape[1], locs2[1,inliers], 'o', c='#00FF00', markerfacecolor='none', linewidth=2)
    for i in range(len(inliers)):
        plt.plot([locs1[0,inliers[i]], locs2[0,inliers[i]]+im1.shape[1]], [locs1[1,inliers[i]], locs2[1,inliers[i]]],
                 c='#00FF00')
    plt.title("Ransac's results")
    plt.show()

def cv_draw_matches(im1, im2, locs1, locs2, inliers):
    im3 = appendimages(im1, im2)
    out1, out2 = np.delete(locs1, inliers, axis=1), np.delete(locs2, inliers, axis=1)
    for i in range(out1.shape[1]):
        cv.circle(im3, (int(round(out1[0,i])),int(round(out1[1,i]))), 5, (0,0,255), 1, cv.LINE_AA)
        cv.circle(im3, (int(round(out2[0,i]+im1.shape[1])), int(round(out2[1,i]))), 5, (0, 0, 255), 1, cv.LINE_AA)
    for i in range(len(inliers)):
        cv.circle(im3, (int(round(locs1[0,inliers[i]])),int(round(locs1[1,inliers[i]]))), 5, (0, 255, 0), 1, cv.LINE_AA)
        cv.circle(im3, (int(round(locs2[0,inliers[i]]+im1.shape[1])), int(round(locs2[1,inliers[i]]))), 5, (0, 255, 0), 1, cv.LINE_AA)
        pt1 = (int(round(locs1[0,inliers[i]])), int(round(locs1[1,inliers[i]])))
        pt2 = (int(round(locs2[0,inliers[i]]+im1.shape[1])), int(round(locs2[1,inliers[i]])))
        cv.line(im3, pt1, pt2, (0,255,0), 1, cv.LINE_AA)
    return im3