import numpy as np


def imageblending(warped_img1, warped_img2, w1, w2):
    mass = w1 + w2
    mass[mass == 0] = np.nan

    output_canvas = np.zeros_like(warped_img1)
    for c in range(3):
        output_canvas[:,:,c] = ((warped_img1[:,:,c] * w1) + (warped_img2[:,:,c] * w2)) / mass
    return output_canvas
