import numpy as np
from scipy import linalg
import math


def normalise2dpts(pts):
    """
    param:
      pts -  3xN array of 2D homogeneous coordinates
    return:
      newpts -  3xN array of transformed 2D homogeneous coordinates
      T      -  The 3x3 transformation matrix, newpts = T*pts
    """
    c = np.mean(pts[:2], axis=1)
    newp = np.ones_like(pts)
    newp[0, :] = pts[0, :] - c[0]
    newp[1, :] = pts[1, :] - c[1]

    dist = np.sqrt(np.square(newp[0, :]) + np.square(newp[1, :]))
    meandist = np.mean(dist)

    scale = math.sqrt(2) / meandist

    T = np.array([[scale, 0, -scale * c[0]],
                  [0, scale, -scale * c[1]],
                  [0, 0, 1]])
    newpts = np.dot(T, pts)
    return newpts, T


def get_conditioner_from_pts(Pts):
    Dim = Pts.shape[0]
    Pts = Pts[0:Dim - 1, :]

    m = np.mean(Pts, axis=1)
    s = np.std(Pts, axis=1, ddof=1)
    s = s + (s == 0)

    norm = math.sqrt(2) / s

    T = np.array([[norm[0], 0, -norm[0] * m[0]],
                  [0, norm[1], -norm[1] * m[1]],
                  [0, 0, 1]])
    return T


def condition_2d(p, C):
    pc = np.dot(C, p)
    return pc


def generate_A(xs1, xs2):
    c = xs1.shape[1]
    A = np.zeros((2 * c, 9))
    ooo = np.zeros(3)
    for k in range(c):
        p1 = xs1[:, k]
        p2 = xs2[:, k]
        A[2 * k] = np.hstack((np.dot(p1, p2[2]), ooo, -np.dot(p1, p2[0])))
        A[2 * k + 1] = np.hstack((ooo, np.dot(p1, p2[2]), -np.dot(p1, p2[1])))
    return A


def homography_fit(x1, x2):
    nf1, N1 = normalise2dpts(x1)
    nf2, N2 = normalise2dpts(x2)
    # condition points
    C1 = get_conditioner_from_pts(nf1)
    C2 = get_conditioner_from_pts(nf2)
    cf1 = condition_2d(nf1, C1)
    cf2 = condition_2d(nf2, C2)

    A = generate_A(cf1, cf2)

    u, s, v = linalg.svd(A)

    h = v[8]
    H = np.reshape(h, (3, 3))
    # decondition
    H = np.linalg.inv(C2).dot(H).dot(C1)
    # denormalise
    H = np.linalg.inv(N2).dot(H).dot(N1)

    return H / H[2, 2]


def homography_res(H, x1, x2):
    x1, T1 = normalise2dpts(x1)
    x2, T2 = normalise2dpts(x2)
    H = T2.dot(H).dot(np.linalg.inv(T1))

    # Calculate, in both directions, the transfered points
    Hx1 = np.dot(H, x1)
    invHx2 = np.linalg.inv(H).dot(x2)

    x1 /= x1[2]
    x2 /= x2[2]
    Hx1 /= Hx1[2]
    invHx2 /= invHx2[2]

    dist = np.sum(np.square(x1 - invHx2), axis=0) + np.sum(np.square(x2 - Hx1), axis=0)

    return dist


def get_hom_final_size(img1, img2, Hg):
    box2 = np.array([[0, img2.shape[1] - 1, img2.shape[1] - 1,                 0],
                     [0,                 0, img2.shape[0] - 1, img2.shape[0] - 1],
                     [1,                 1,                 1,                 1]])
    box2_ = linalg.solve(Hg, box2)
    box2_[0, :] = box2_[0, :] / box2_[2, :]
    box2_[1, :] = box2_[1, :] / box2_[2, :]

    min_x = min(min(box2_[0, :]), 0)
    max_x = max(max(box2_[0, :]), img1.shape[1]-1)
    min_y = min(min(box2_[1, :]), 0)
    max_y = max(max(box2_[1, :]), img1.shape[0]-1)

    return min_x, max_x, min_y, max_y
