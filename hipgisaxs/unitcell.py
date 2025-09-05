from abc import ABC

try:
    import cupy as np
except ImportError:
    import numpy as np

import warnings

from .ff import cuboid, cone, cone_stack, cone_shell, cylinder, sphere

try:
    from .ff import MeshFF
except ImportError:
    warnings.warn("failed to import meshff, required for triangulated structures", stacklevel=2)


def makeShapeObject(shape) -> "ShapeBase":
    fftype = shape["formfactor"]
    if fftype in globals():
        shape_args = {k: v for k, v in shape.items() if k != "formfactor"}
        ob = globals()[fftype](**shape_args)
    else:
        raise ValueError("Unknown formfactor")
    return ob


class ShapeBase(ABC):
    def __init__(self, delta, beta, locations=None, orient=None):
        self.delta = delta
        self.beta = beta
        self.locations = locations
        self.orient = orient

        self.n = complex(1 - self.delta, self.beta)
        self.n2 = self.n**2

    def ff(self, qx, qy, qz):
        pass


class CoreShell(ShapeBase):
    def __init__(self, core, shell):
        self.core = makeShapeObject(core)
        self.shell = makeShapeObject(shell)
        super().__init__(
            delta=self.shell.delta,
            beta=self.shell.beta,
            locations=self.shell.locations,
            orient=self.shell.orient,
        )

    def ff(self, qx, qy, qz, n_ref=complex(1, 0)):
        ff_core = self.core.ff(qx, qy, qz)
        ff_shell = self.shell.ff(qx, qy, qz)
        contrast_shell = self.shell.n2 - n_ref**2
        contrast_core = self.core.n2 - self.shell.n2
        return contrast_shell * ff_shell + contrast_core * ff_core


class Unitcell:
    def __init__(self, shapes, delta=0, beta=0):
        self.shapes: list[ShapeBase] = []
        self.delta = delta
        self.beta = beta

        self.n = complex(1 - delta, beta)
        self.n2 = self.n**2

        for shape in shapes:
            if shape["formfactor"] == "CoreShell":
                self.shapes.append(CoreShell(shape["Core"], shape["Shell"]))
            else:
                self.shapes.append(makeShapeObject(shape))

    def ff(self, qx, qy, qz):
        ff = np.zeros(qx.size, dtype=complex)
        for shape in self.shapes:
            if isinstance(shape, CoreShell):
                shape_ff = shape.ff(qx, qy, qz, n_ref=self.n)
            else:
                contrast_particle = shape.n2 - self.n2
                shape_ff = contrast_particle * shape.ff(qx, qy, qz)

            locs = shape.locations
            if locs is None:
                locs = [{"x": 0, "y": 0, "z": 0}]

            tempff = np.zeros(qx.size, dtype=complex)
            for loc in locs:
                phase = np.exp(1j * (qx * loc["x"] + qy * loc["y"] + qz * loc["z"]))
                tempff += phase

            ff += shape_ff * tempff

        return ff


# ----basic shapes------#
class Cylinder(ShapeBase):
    def __init__(self, *args, radius, height, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.height = height

    # calculate form-factor
    def ff(self, qx, qy, qz):
        return cylinder(qx, qy, qz, self.radius, self.height, self.orient)


class Cuboid(ShapeBase):
    def __init__(self, *args, length, width, height, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = length
        self.width = width
        self.height = height

    # calculate form-factor
    def ff(self, qx, qy, qz):
        return cuboid(qx, qy, qz, self.length, self.width, self.height, self.orient)


class Cone(ShapeBase):
    def __init__(self, *args, radius, height, angle, ndeg=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.height = height
        self.angle = angle
        self.ndeg = ndeg

    # calculate form-factor
    def ff(self, qx, qy, qz):
        return cone(qx, qy, qz, self.radius, self.height, self.angle, self.ndeg, self.orient)


class Sphere(ShapeBase):
    def __init__(self, *args, radius, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius

    # calculate form-factor
    def ff(self, qx, qy, qz):
        return sphere(qx, qy, qz, self.radius)


"""
class MeshFT:
    def __init__(self):
        pass

    def ff(self, qx, qy, qz):
        from stl import mesh
        mesh = mesh.Mesh.from_file(self.meshfile)
        vertices = mesh.vectors.astype(float)
        qz = qz.astype(complex)
        rot = np.eye(3)
        return MeshFF(qx, qy, qz, rot, vertices)
"""
