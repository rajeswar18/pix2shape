import numpy as np
from diffrend.numpy.vector import Vector

class Quaternion(object):
    """
    Usage:
    1. Rotate a vector p about an axis q by angle theta:
        Quaternion(angle=theta, axis=q).R * p  [Here p is in homogeneous coordinates and can be 4 x N matrix]
    2.
    """

    def __init__(self, coeffs=None, angle=None, axis=None, vector=None):
        assert (coeffs is not None) or (angle is not None and axis is not None) or vector is not None
        if coeffs is not None:
            self.coeffs = np.array(coeffs)

        if vector is not None:
            self.coeffs = np.array([0, vector[0], vector[1], vector[2]])

        if angle is not None and axis is not None:
            assert coeffs is None
            cos_theta = np.cos(angle / 2.)
            sin_theta = np.sin(angle / 2.)
            if isinstance(axis, Vector):
                axis = axis.normalize()
            else:
                axis = np.array(axis)[:3]
                axis = axis / np.sqrt(np.sum(axis ** 2))
            v = axis * sin_theta
            self.coeffs = np.append(np.array([cos_theta]), v)

    def __str__(self):
        return str(list(self.coeffs))

    def __add__(self, other):
        return Quaternion(self.coeffs + other.coeffs)

    def __sub__(self, other):
        return Quaternion(self.coeffs - other.coeffs)

    def __mul__(self, other):
        return Quaternion(np.dot(self.matrix, other.coeffs))

    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self.__mul__(~other)
        return Quaternion(self.coeffs / other)

    def __invert__(self):
        return self.conj / self.norm_sqr

    def __getitem__(self, item):
        return self.coeffs[item]

    @property
    def conj(self):
        return Quaternion(self.coeffs * [1, -1, -1, -1])

    @property
    def matrix(self):
        a, b, c, d = self.coeffs
        return np.array([[a, -b, -c, -d],
                         [b, a, -d, c],
                         [c, d, a, -b],
                         [d, -c, b, a]])

    @property
    def norm_sqr(self):
        return np.sum(self.coeffs ** 2)

    @property
    def norm(self):
        return np.sqrt(self.norm_sqr)

    @property
    def rotation_matrix(self):
        """http://www.cs.ucr.edu/~vbz/resources/quatut.pdf (pg. 6, 7)
        :return:
        """
        w, x, y, z = self.coeffs
        s = 2. / self.norm_sqr
        # For multiplication with homogeneous coordinates
        return np.array([[1 - s * (y**2 + z ** 2), s * (x * y - w * z), s * (x * z + w * y), 0],
                        [s * (x * y + w * z), 1 - s * (x ** 2 + z ** 2), s * (y * z - w * x), 0],
                        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x ** 2 + y ** 2), 0],
                         [0, 0, 0, 1]])

    @property
    def R(self):
        """Returns angle-axis rotation matrix
        :return:
        """
        return self.rotation_matrix

    def rotate(self, angle_rad, axis):
        """
        :param angle_rad: Angle of rotation in radians
        :param axis: Axis of rotation as a 3D vector (numpy.array) or list, or a Quaternion
        :return:
        """
        q = Quaternion(angle=angle_rad, axis=axis)
        return q * self.__mul__(~q)

    def rotate_deg(self, angle_deg, axis):
        return self.rotate(angle_deg * np.pi / 180., axis)



def test():
    a = Quaternion([1, 2, 3, 4])
    b = Quaternion([3, 4, 5, 6])
    ab = a * b
    a_div_b = a / b

    print(a)
    print(b)
    print(ab)
    print(a_div_b)


if __name__ == '__main__':
    test()


# a = Quaternion(angle=0, axis=[0, 0, -1])
