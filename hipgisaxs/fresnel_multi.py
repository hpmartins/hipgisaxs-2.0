import numpy as np


class Layer:
    def __init__(self, delta, beta, order, thickness):
        self.one_minus_n2 = 2 * complex(delta, beta)
        self.order = order
        self.thickness = thickness
        self.zval = 0


class MultiLayer:
    def __init__(self):
        self.layers = [Layer(0, 0, 0, 0)]
        self.substrate = Layer(4.88e-6, 7.37e-08, -1, 0)
        self._setup_ = False

    def insert(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("only Layer types can be inserted into multilayered object")
        if not layer.order > 0:
            raise ValueError("the order of layer must be greater than 0")
        self.layers.insert(layer.order, layer)

    def setup_multilayer(self):
        if self._setup_:
            return

        if len(self.layers) > 1 and self.layers[1].one_minus_n2 == 0:
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

    def parratt_recursion(self, alpha, wavelength, order):
        self.setup_multilayer()
        nlayer = len(self.layers)
        shape = np.shape(alpha)
        # account for scalar case
        if len(shape) == 0:
            shape = (1,)

        # sin(alpha)
        sin_a = np.sin(alpha)

        # initialize
        dims = (nlayer - 1,) + shape
        dim2 = (nlayer,) + shape

        # cacl k-value
        k0 = 2 * np.pi / wavelength
        kz = np.zeros(dim2, np.complex128)
        for i in range(nlayer):
            kz[i, :] = -k0 * np.sqrt(sin_a**2 - self.layers[i].one_minus_n2)

        # calculate Rs
        R = np.zeros(dim2, dtype=np.complex128)
        T = np.zeros(dim2, dtype=np.complex128)
        T[-1] = 1
        for i in reversed(range(nlayer - 1)):
            z = self.layers[i].zval
            en = np.exp(-1j * kz[i] * z)
            ep = np.exp(1j * kz[i] * z)
            t0 = (kz[i] + kz[i + 1]) / (2 * kz[i])
            t1 = (kz[i] - kz[i + 1]) / (2 * kz[i])
            T[i] = T[i + 1] * en * t0 + R[i + 1] * en * t1
            R[i] = T[i + 1] * ep * t1 + R[i + 1] * ep * t0

        T0 = T[0]

        return T[order] / T0, R[order] / T0

    def propagation_coeffs(self, alphai, alpha, wavelength, order):
        Ti, Ri = self.parratt_recursion(alphai, wavelength, order)

        fc = np.zeros((4,) + alpha.shape, np.complex128)
        mask = alpha > 0
        if np.any(mask):
            alpha_positive = alpha[mask]
            Tf, Rf = self.parratt_recursion(alpha_positive, wavelength, order)

            fc[0, mask] = Ti * Tf
            fc[1, mask] = Ri * Tf
            fc[2, mask] = Ti * Rf
            fc[3, mask] = Ri * Rf

        return fc
