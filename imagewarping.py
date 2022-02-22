import numpy as np
from scipy import linalg
import cv2 as cv


def _meshgrid(width_max, width_min, height_max, height_min):
    width = int(round(width_max - width_min + 1))
    height = int(round(height_max - height_min + 1))
    x_t = np.matmul(np.ones([height, 1]),
                    np.transpose(np.expand_dims(np.linspace(width_min, width_max, width), 1), [1, 0]))
    y_t = np.matmul(np.expand_dims(np.linspace(height_min, height_max, height), 1),
                    np.ones([1, width]))

    x_t_flat = np.reshape(x_t, (1, -1))
    y_t_flat = np.reshape(y_t, (1, -1))

    ones = np.ones_like(x_t_flat)
    grid = np.vstack((x_t_flat, y_t_flat, ones))
    return grid

def transform(Hg, width_max, width_min, height_max, height_min):
    out_width = int(round(width_max - width_min + 1))
    out_height = int(round(height_max - height_min + 1))

    grid = _meshgrid(width_max, width_min, height_max, height_min)

    T_g = np.matmul(Hg, grid)

    x_ = T_g[0] / T_g[2]
    y_ = T_g[1] / T_g[2]

    x_map = x_.reshape([out_height, out_width])
    y_map = y_.reshape([out_height, out_width])

    return x_map.astype(np.float32), y_map.astype(np.float32)


def _meshgrid2(width_max, width_min, height_max, height_min, sh, eh, sw, ew):
    hn = eh - sh + 1
    wn = ew - sw + 1

    width = int(round(width_max - width_min + 1))
    height = int(round(height_max - height_min + 1))
    x_t = np.matmul(np.ones([hn, 1]),
                    np.transpose(np.expand_dims(np.linspace(width_min, width_max, width)[sw:sw+wn], 1), [1, 0]))
    y_t = np.matmul(np.expand_dims(np.linspace(height_min, height_max, height)[sh:sh+hn], 1),
                    np.ones([1, wn]))

    x_t_flat = np.reshape(x_t, (1, -1))
    y_t_flat = np.reshape(y_t, (1, -1))

    ones = np.ones_like(x_t_flat)
    grid = np.vstack((x_t_flat, y_t_flat, ones))
    return grid

def transform3(Hmdlt, width_max, width_min, height_max, height_min, grid_w, grid_h):

    Hmdlt = Hmdlt.reshape(grid_h+1, grid_w+1, 3, 3)

    out_width = int(round(width_max - width_min + 1))
    out_height = int(round(height_max - height_min + 1))

    gh = int(out_height / grid_h)
    gw = int(out_width / grid_w)

    x_ = []
    y_ = []
    for i in range(grid_h):
        row_x_ = []
        row_y_ = []
        for j in range(grid_w):
            H = Hmdlt[i, j, :, :]
            sh = i * gh
            eh = (i + 1) * gh - 1
            sw = j * gw
            ew = (j + 1) * gw - 1
            if (i == grid_h - 1):
                eh = out_height - 1
            if (j == grid_w - 1):
                ew = out_width - 1

            grid = _meshgrid2(width_max, width_min, height_max, height_min, sh, eh, sw, ew)

            T_g = linalg.solve(H, grid)
            x_s_flat = T_g[0]
            y_s_flat = T_g[1]
            z_s_flat = T_g[2]

            t_1 = np.ones_like(z_s_flat)
            t_0 = np.zeros_like(z_s_flat)

            sign_z_flat = np.where(z_s_flat >= 0, t_1, t_0) * 2 - 1
            z_s_flat = z_s_flat + sign_z_flat * 1e-8
            x_s_flat = x_s_flat / z_s_flat
            y_s_flat = y_s_flat / z_s_flat

            x_s = np.reshape(x_s_flat, [eh - sh + 1, ew - sw + 1])
            y_s = np.reshape(y_s_flat, [eh - sh + 1, ew - sw + 1])
            row_x_.append(x_s)
            row_y_.append(y_s)
        row_x = np.hstack(row_x_)
        row_y = np.hstack(row_y_)
        x_.append(row_x)
        y_.append(row_y)

    x_map = np.vstack(x_).reshape([out_height, out_width])
    y_map = np.vstack(y_).reshape([out_height, out_width])

    return x_map.astype(np.float32), y_map.astype(np.float32)


def imagewarping(img1, img2, H, min_x, max_x, min_y, max_y, C1=None, C2=None):

    out_width = int(round(max_x - min_x + 1))
    out_height = int(round(max_y - min_y + 1))
    off = [int(round(-min_x)), int(round(-min_y))]
    warped_img1 = np.zeros((out_height, out_width, 3), np.uint8)
    warped_img1[off[1]:off[1] + img1.shape[0], off[0]:off[0] + img1.shape[1], :] = img1
    mask1 = np.ones((img1.shape[0], img1.shape[1]))
    warped_mask1 = np.zeros([out_height, out_width])
    warped_mask1[off[1]:off[1] + img1.shape[0], off[0]:off[0] + img1.shape[1]] = mask1

    if C1 != None and C2 != None:    # mdlt
        map_x, map_y = transform3(H, max_x, min_x, max_y, min_y, C1 - 1, C2 - 1)
        warped_img2 = cv.remap(img2, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)

    else:    # global
        map_x, map_y = transform(H, max_x, min_x, max_y, min_y)
        warped_img2 = cv.remap(img2, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)

    mask2 = np.ones((img2.shape[0], img2.shape[1]))
    warped_mask2 = cv.remap(mask2, map_x, map_y, interpolation=cv.INTER_LINEAR)

    return warped_img1, warped_img2, warped_mask1, warped_mask2





