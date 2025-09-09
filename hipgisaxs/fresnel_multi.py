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

    def parratt_recursion(self, alpha, k0, order=0) -> tuple[np.ndarray, np.ndarray]:
        self.setup_multilayer()
        nlayer = len(self.layers)
        shape = np.shape(alpha)
        # account for scalar case
        if len(shape) == 0:
            shape = (1,)
            alpha = np.array([alpha])

        n2_layers = np.array([layer.n2 for layer in self.layers])
        n2_layers = n2_layers[:, np.newaxis]

        cos2_alpha = np.cos(alpha) ** 2
        cos2_alpha = cos2_alpha[np.newaxis, :]

        kz = k0 * np.sqrt(n2_layers - cos2_alpha)

        t_coeff = np.zeros((nlayer - 1,) + shape, dtype=np.complex128)
        r_coeff = np.zeros((nlayer,) + shape, dtype=np.complex128)
        r_coeff[-1] = 0.0  # no reflection coming from substrate
        for i in reversed(range(nlayer - 1)):
            kz_ratio = kz[i + 1] / kz[i]
            slp = 1.0 + kz_ratio  # (kz[i] + kz[i + 1])/kz[i]
            slm = 1.0 - kz_ratio  # (kz[i] - kz[i + 1])/kz[i]

            z = self.layers[i].thickness
            phase = np.exp(1j * kz[i] * z) if np.isfinite(z) else 1
            propagation = phase / (slp + slm * r_coeff[i + 1])

            t_coeff[i] = 2.0 * propagation
            r_coeff[i] = phase * (slm + slp * r_coeff[i + 1]) * propagation
            # first iteration:
            # t_coeff = 2/slp = 2kz[i]/(kz[i] + kz[i + 1]) = t_i_i1
            # r_coeff = slm/slp = (kz[i] - kz[i + 1])/(kz[i] + kz[i + 1]) = r_i_i1

        T = np.zeros((nlayer,) + shape, dtype=np.complex128)
        R = np.zeros((nlayer,) + shape, dtype=np.complex128)

        T[0] = 1.0
        R[0] = r_coeff[0] * T[0]
        for i in range(1, nlayer):
            T[i] = T[i - 1] * t_coeff[i - 1]
            R[i] = r_coeff[i] * T[i]

        return T[order], R[order]

    def propagation_coeffs(self, alphai, alpha, k0, order):
        Ti, Ri = self.parratt_recursion(alphai, k0, order)
        Tf, Rf = self.parratt_recursion(alpha, k0, order)

        result_shape = np.broadcast(Ti[:, np.newaxis], Tf).shape
        fc = np.zeros((4,) + result_shape, complex)

        fc[0] = Ti[:, np.newaxis] * Tf
        fc[1] = Ri[:, np.newaxis] * Tf
        fc[2] = Ti[:, np.newaxis] * Rf
        fc[3] = Ri[:, np.newaxis] * Rf

        return fc
