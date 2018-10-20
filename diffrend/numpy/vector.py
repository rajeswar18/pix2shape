import numpy as np


class Vector(object):
    def __init__(self, v):
        assert type(v) is np.ndarray or type(v) is list
        self.v = np.array(v)

    def __str__(self):
        return str(self.v)

    def __add__(self, other):
        if type(other) is np.ndarray:
            return self.v + other
        elif isinstance(other, Vector):
            return Vector(self.v + other.v)
        else:
            raise ValueError('Invalid type')

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(np.dot(self.v, other.v))
        return Vector(self.v * other)

    def __getitem__(self, item):
        return self.v[item]

    @property
    def cross_matrix(self):
        x, y, z = self.v
        return np.array([[0, -z, y],
                         [z, 0, -x],
                         [-y, x, 0]])

    @property
    def norm(self):
        return self.normalize()

    def normalize(self):
        return Vector(self.v / np.sqrt(np.sum(self.v ** 2)))


class Ray(object):
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=np.float32)
        self.direction = np.array(direction, dtype=np.float32)
        self.direction /= np.sqrt(np.sum(self.direction ** 2))

    def __str__(self):
        return 'origin: ' + str(self.origin) + ' direction: ' + str(self.direction)

    def point(self, t):
        return self.direction * t + self.origin

# r = Ray([0, 0, 0], [1, 0, 0])
# print(r)
# a = r.point(10)
# print(a)
