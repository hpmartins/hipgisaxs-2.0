import numpy as np

from ..rotation import rotate


def cuboid(qx, qy, qz, length, width, height, orientation=None):

    if orientation is None:
        q1, q2, q3 = qx.ravel(), qy.ravel(), qz.ravel()
    else:
        q1, q2, q3 = rotate(qx, qy, qz, orientation)

    volume = length * width * height
    z_shift = np.exp(1j * q3 * height / 2)

    arg1 = q1 * (length / 2) / np.pi
    arg2 = q2 * (width / 2) / np.pi
    arg3 = q3 * (height / 2) / np.pi

    return volume * z_shift * np.sinc(arg1) * np.sinc(arg2) * np.sinc(arg3)
