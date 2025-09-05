import numpy as np
from .rotation import rotate


def laue_function(x, N):
    numerator = np.sin(N * x / 2)
    denominator = np.sin(x / 2)
    result = np.where(np.abs(N * x) < 1e-16, N, numerator / denominator)
    return result


def structure_factor(q1, q2, q3, d_space, numelm, orient=None):

    if not np.shape(d_space) == (3, 3):
        raise TypeError("d_space must be a 3x3 matrix")

    if not np.shape(numelm)[0] == 3:
        raise TypeError("numelm must be a vector of 3")

    # rotate the q-vectors
    if orient is None:
        qx, qy, qz = q1.ravel(), q2.ravel(), q3.ravel()
    else:
        qx, qy, qz = rotate(q1, q2, q3, orient)

    sf = np.ones_like(qx, dtype=np.complex128)
    for i, v in enumerate(d_space):
        if numelm[i] > 1:
            qd = qx * v[0] + qy * v[1] + qz * v[2]
            sf *= laue_function(qd, numelm[i]) / numelm[i]

    return sf
