#! /usr/bin/env python

import numpy as np


def sphere(qx, qy, qz, radius):
    vol = (4 / 3) * np.pi * radius**3
    qR = np.sqrt(qx**2 + qy**2 + qz**2) * radius
    tmp = np.exp(1j * qz * radius)

    ff = np.ones_like(qR) * vol
    mask = qR != 0

    term = np.sin(qR[mask]) - qR[mask] * np.cos(qR[mask])
    ff[mask] = (3 * vol * tmp[mask] * term) / qR[mask] ** 3

    return ff
