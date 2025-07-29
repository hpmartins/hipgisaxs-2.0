import numpy as np
from scipy.special import j1
from ..rotation import rotate


def cylinder(qx, qy, qz, radius, height, orientation=None):
    if orientation is None:
        q1, q2, q3 = qx.ravel(), qy.ravel(), qz.ravel()
    else:
        q1, q2, q3 = rotate(qx, qy, qz, orientation)

    vol = np.pi * radius**2 * height
    qpR = np.sqrt(q1**2 + q2**2) * radius
    qparH = q3 * height / 2

    f1 = np.sin(qparH) / qparH
    f2 = j1(qpR) / qpR
    f3 = np.exp(1j * height * q3 / 2)

    return 2 * vol * f1 * f2 * f3
