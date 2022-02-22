import numpy as np
from scipy import linalg
from homography import normalise2dpts, get_conditioner_from_pts, condition_2d, generate_A


class APAP_stitching():
    def __init__(self, gamma, sigma):
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma

    def __call__(self, src_p, dst_p, vertices):
        """
        local homography estimation
        :param src_p: shape [3, N]
        :param dst_p: shape [3, N]
        :param vertices: shape [mesh_size*mesh_size, 2]
        :return: np.ndarray [meshsize*meshsize, 9]
        """
        nf1, N1 = normalise2dpts(src_p)
        nf2, N2 = normalise2dpts(dst_p)
        C1 = get_conditioner_from_pts(nf1)
        C2 = get_conditioner_from_pts(nf2)
        cf1 = condition_2d(nf1, C1)
        cf2 = condition_2d(nf2, C2)

        A = generate_A(cf1, cf2)
        # print('A', A.shape)

        Kp = src_p[:2].T  # ->shape [N, 2]
        # print('Kp', Kp.shape)

        Hmdlt = np.zeros((len(vertices), 9))
        for i in range(len(vertices)):
            # Obtain kernel
            dist = np.sqrt(np.sum(np.square(vertices[i, :] - Kp), axis=1))
            Gki = np.exp(-dist / self.sigma ** 2)

            # Capping/offsetting kernel
            Wi = np.maximum(Gki, self.gamma)

            W = np.diag(np.repeat(Wi, 2))
            WA = np.dot(W, A)

            U, S, V = linalg.svd(WA, 0)
            v = V[8]

            h = v.reshape(3, 3)

            # De-condition
            h = np.linalg.inv(C2).dot(h).dot(C1)
            # De-normalize
            h = np.linalg.inv(N2).dot(h).dot(N1)
            h = h / h[2, 2]

            Hmdlt[i, :] = h.ravel()

        return Hmdlt


def get_mdlt_final_size(img1, img2, Hmdlt, C1, C2):
    x_list = np.linspace(0, img2.shape[1]-1, C1)
    y_list = np.linspace(0, img2.shape[0]-1, C2)
    x, y = np.meshgrid(x_list, y_list)

    # out_x = np.zeros(x.shape)
    # out_y = np.zeros(y.shape)
    out = []

    for i in range(y.shape[0]):
        for j in range(x.shape[1]):
            if i==0 or i==y.shape[0]-1 or j==0 or j==x.shape[1]-1:    # boundaries
                grididx = i * x.shape[1] + j

                T = Hmdlt[grididx, :].reshape(3, 3)
                # T = linalg.inv(T)

                in_x = x[i, j]
                in_y = y[i, j]

                out_x = (T[0, 0] * in_x + T[0, 1] * in_y + T[0, 2]) / \
                        (T[2, 0] * in_x + T[2, 1] * in_y + T[2, 2])

                out_y = (T[1, 0] * in_x + T[1, 1] * in_y + T[1, 2]) / \
                        (T[2, 0] * in_x + T[2, 1] * in_y + T[2, 2])

                out.append([out_x, out_y])

    out = np.array(out)

    min_x = min(min(out[:, 0]), 0)
    max_x = max(max(out[:, 0]), img1.shape[1]-1)
    min_y = min(min(out[:, 1]), 0)
    max_y = max(max(out[:, 1]), img1.shape[0]-1)

    return min_x, max_x, min_y, max_y

