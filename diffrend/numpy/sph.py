import numpy as np
from diffrend.utils.lightprobe import load_lightprobe


def RealSH9_cart(X, Y, Z):
    """
    :param X:
    :param Y:
    :param Z:
    :return: meshgrid or vector with the last dim being 9 elements
    corresponding to the first 9 coefficients:return:
    """
    # 0: (0, 0), 1:(1,-1), 2:(1, 0), ...
    sqrt_1_over_pi = np.sqrt(1 / np.pi)
    sqrt_3_over_pi = np.sqrt(3 / np.pi)
    sqrt_5_over_pi = np.sqrt(5 / np.pi)
    sqrt_15_over_pi = np.sqrt(15 / np.pi)

    Y0 = 1 / 2. * sqrt_1_over_pi * np.ones_like(X)  # Y(0, 0)
    Y1 = 1/2. * sqrt_3_over_pi * Y  # Y(1, -1)
    Y2 = 1/2. * sqrt_3_over_pi * Z  # Y(1, 0)
    Y3 = 1/2. * sqrt_3_over_pi * X  # Y(1, 1)
    Y4 = 1/2. * sqrt_15_over_pi * X * Y  # Y(2, -2)
    Y5 = 1/2. * sqrt_15_over_pi * Y * Z  # Y(2, -1)
    Y6 = 1/4. * sqrt_5_over_pi * (3 * Z ** 2 - 1)  # Y(2, 0)
    Y7 = 1/2. * sqrt_15_over_pi * X * Z  # Y(2, 1)
    Y8 = 1/4. * sqrt_15_over_pi * (X ** 2 - Y ** 2)  # Y(2, 2)

    return np.stack((Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8), axis=-1)


def RealSH9_polar(theta, phi):
    """
    :param theta: vector or meshgrid
    :param phi:
    :return: meshgrid or vector with the last dim being 9 elements
    corresponding to the first 9 coefficients
    """
    X = np.cos(phi) * np.sin(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(theta)

    return RealSH9_cart(X, Y, Z)


def radiance_SH9(envmap):
    """
    :param envmap: H x W x C
    :return: C x 9
    """
    if type(envmap) is str:
        envmap, _ = load_lightprobe(envmap)

    H, W, C = envmap.shape
    i, j = np.mgrid[0:H, 0:W]

    v = (W / 2.0 - i) / (W / 2.0)
    u = (j - H / 2.0) / (H / 2.0)
    r = np.sqrt(u ** 2 + v ** 2)

    valid_region_mask = r <= 1.0
    envmap = envmap * valid_region_mask[..., np.newaxis]

    theta = np.pi * r
    phi = np.arctan2(v, u)

    domega = (2 * np.pi / W) * (2 * np.pi / H) * np.sinc(theta)

    Y = RealSH9_polar(theta, phi)

    # simpler op for channel c
    # L_c = np.sum(np.sum(envmap[..., c][..., np.newaxis] * Y * domega[..., np.newaxis], axis=0), axis=0)
    # Operate on all channels
    L = np.sum(np.sum(envmap[..., np.newaxis] * Y[..., np.newaxis, :] * domega[..., np.newaxis, np.newaxis], axis=0),
               axis=0)
    # {'coeffs': L, 'valid': valid_region_mask, 'sh': Y}
    return L


def lm2idx(l, m):
    """ Spherical Harmonics (l, m) to linear index
    :param l: 0-based
    :param m:
    :return: 0-based index
    """
    return l ** 2 + (l + m)


def RealSH9_ZonalMatrix(L):
    """ Eq. 12 of Ramamoorthi's envmap paper
    :param L: 9D real harmonic coefficients
    :return: 4 x 4 matrix
    """
    c1 = 0.429043
    c2 = 0.511664
    c3 = 0.743125
    c4 = 0.886227
    c5 = 0.247708
    M = []
    for ch in range(L.shape[0]):
        L00 = L[ch, lm2idx(0, 0)]
        L1m1 = L[ch, lm2idx(1, -1)]
        L10 = L[ch, lm2idx(1, 0)]
        L11 = L[ch, lm2idx(1, 1)]
        L2m2 = L[ch, lm2idx(2, -2)]
        L2m1 = L[ch, lm2idx(2, -1)]
        L20 = L[ch, lm2idx(2, 0)]
        L21 = L[ch, lm2idx(2, 1)]
        L22 = L[ch, lm2idx(2, 2)]

        M.append(np.array([[c1 * L22, c1 * L2m2, c1 * L21, c2 * L11],
                      [c1 * L2m2, -c1 * L22, c1 * L2m1, c2 * L1m1],
                      [c1 * L21, c1 * L2m1, c3 * L20, c2 * L10],
                      [c2 * L11, c2 * L1m1, c2 * L10, c4 * L00 - c5 * L20]
                      ]))
    return np.stack(M, axis=-1)


def irradiance(M, normal):
    """
    :param M:
    :param normal: N x 4
    :return:
    """
    assert normal.shape[-1] == 3
    init_dim = M.ndim
    if init_dim == 2:
        M = M[..., np.newaxis]
    b = []
    for ch in range(M.shape[-1]):
        a = (np.matmul(normal, M[:3, :, ch]) + M[3, :, ch])
        b.append(np.sum(a[..., :3] * normal, axis=-1) + a[..., 3])
    if init_dim == 2:
        b = b[0]
    else:
        b = np.stack(b, axis=-1)

    return b


def irradiance_polar(M, theta, phi):
    X = np.cos(phi) * np.sin(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(theta)

    normal = np.stack((X, Y, Z), axis=-1)
    return irradiance(M, normal)


def reconstruct_SH9(L, dim):
    H, W = dim
    i, j = np.mgrid[0:H, 0:W]

    v = (W / 2.0 - i) / (W / 2.0)
    u = (j - H / 2.0) / (H / 2.0)
    r = np.sqrt(u ** 2 + v ** 2)

    theta = np.pi * r
    phi = np.arctan2(v, u)

    Y = RealSH9_polar(theta, phi)
    domega = (2 * np.pi / W) * (2 * np.pi / H) * np.sinc(theta)
    recon = np.sum(L[np.newaxis, np.newaxis, ...] * Y[..., np.newaxis, :] * domega[..., np.newaxis, np.newaxis],
                   axis=-1)
    return recon


def irrad_Z(L, dim):
    H, W = dim
    i, j = np.mgrid[0:H, 0:W]

    v = (W / 2.0 - i) / (W / 2.0)
    u = (j - H / 2.0) / (H / 2.0)
    r = np.sqrt(u ** 2 + v ** 2)

    theta = np.pi * r
    phi = np.arctan2(v, u)

    M = RealSH9_ZonalMatrix(L)

    return irradiance_polar(M, theta, phi)


def plot_RealSH9(dim):
    H, W = dim
    i, j = np.mgrid[0:H, 0:W]

    v = (W / 2.0 - i) / (W / 2.0)
    u = (j - H / 2.0) / (H / 2.0)
    r = np.sqrt(u ** 2 + v ** 2)

    valid_region = (r <= 1.0)

    theta = np.pi * r
    phi = np.arctan2(v, u)

    Y = RealSH9_polar(theta, phi)
    plt.figure()
    for l in range(3):
        for m in range(-l, l + 1):
            idx = lm2idx(l, m)
            plt_idx = l * 5 + (m + 3)
            plt.subplot(3, 5, plt_idx)
            plt.imshow(Y[..., idx] * valid_region)
            plt.axis('off')


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from data import DIR_DATA

    parser = argparse.ArgumentParser(usage='sph.py --fname lightprobe_path')
    parser.add_argument('--fname', type=str, default=DIR_DATA + '/envmap/rnl_probe.float')
    args = parser.parse_args()

    # rnl_probe.float, grace_probe.float, stpeters_probe.float
    envmap, _ = load_lightprobe(args.fname)
    L = radiance_SH9(envmap)

    H, W, C = envmap.shape
    start = int(H / 2) - 5
    end = int(H / 2) + 5
    print(start, end)
    print(envmap[start:end, start:end, 0])
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 1000), np.linspace(0, 2 * np.pi, 1000))
    R = RealSH9_polar(theta, phi)

    # from mpl_toolkits.mplot3d import axes3d
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    plot_RealSH9((200, 200))
    im = irrad_Z(L, (500, 500))

    plt.figure()
    plt.imshow(im - im.min())

    plt.ioff()
    plt.show()
