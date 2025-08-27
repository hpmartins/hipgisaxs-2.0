import numpy as np


class Layer:
    def __init__(self, delta, beta, order, thickness):
        self.delta = delta
        self.beta = beta
        self.order = order
        self.thickness = thickness
        self.zval = 0
        self.n = complex(1 - delta, beta)
        self.n2 = self.n**2


class MultiLayer:
    def __init__(self):
        self.layers = [Layer(0, 0, 0, 0)]
        self.substrate = Layer(4.88e-6, 7.37e-08, -1, 0)
        self._setup_ = False

    def setup_multilayer(self):
        if self._setup_:
            return

        # Remove duplicate vacuum layer if present
        if len(self.layers) > 1 and self.layers[1].n == 1:
            self.layers = self.layers[1:]
            self.layers[0].thickness = 0
            for idx, layer in enumerate(self.layers):
                layer.order = idx

        # put substrate at the end
        self.layers.append(self.substrate)

        # calc z of every interface
        nlayer = len(self.layers)
        self.layers[0].zval = 0
        for i in range(1, nlayer - 1):
            self.layers[i].zval = self.layers[i - 1].zval + self.layers[i].thickness
        self.layers[-1].zval = np.inf

        # run only once
        self._setup_ = True

    def parratt_recursion(self, alpha, k0, order=0):
        self.setup_multilayer()
        nlayer = len(self.layers)
        shape = np.shape(alpha)
        # account for scalar case
        if len(shape) == 0:
            shape = (1,)

        # sin(alpha)
        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)

        # initialize
        dim2 = (nlayer,) + shape

        # cacl k-value
        kz = np.zeros(dim2, np.complex128)
        kz[0, :] = k0 * sin_a
        for i in range(1, nlayer):
            kz_temp = k0 * np.sqrt(self.layers[i].n2 - cos_a**2)
            kz[i, :] = kz_temp * np.where(np.imag(kz_temp) >= 0, 1, -1)

        # calculate Rs
        R = np.zeros(dim2, dtype=np.complex128)
        T = np.zeros(dim2, dtype=np.complex128)
        T[-1] = 1
        for i in reversed(range(nlayer - 1)):
            z = self.layers[i].thickness
            en = np.exp(-1j * kz[i] * z)
            ep = np.exp(1j * kz[i] * z)
            t0 = (kz[i] + kz[i + 1]) / (2 * kz[i])
            t1 = (kz[i] - kz[i + 1]) / (2 * kz[i])
            T[i] = T[i + 1] * en * t0 + R[i + 1] * en * t1
            R[i] = T[i + 1] * ep * t1 + R[i + 1] * ep * t0

        T0 = T[0]

        return T[order] / T0, R[order] / T0

    def propagation_coeffs(self, alphai, alpha, k0, order):
        Ti, Ri = self.parratt_recursion(alphai, k0, order)
        Tf, Rf = self.parratt_recursion(alpha, k0, order)

        result_shape = np.broadcast(Ti, Tf).shape
        fc = np.zeros((4,) + result_shape, complex)

        fc[0] = Ti * Tf
        fc[1] = Ri * Tf
        fc[2] = Ti * Rf
        fc[3] = Ri * Rf

        return fc
