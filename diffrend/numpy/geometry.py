import numpy as np
import abc
from diffrend.numpy.vector import Vector, Ray

class SceneObject(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, pos, material=None):
        self.pos = pos
        self.material = material

    @abc.abstractmethod
    def intersection(self, ray):
        pass
    
    # @abc.abstractmethod
    # @property
    # def vertices(self):
    #     pass
    #
    # @abc.abstractmethod
    # @property
    # def indices(self):
    #     pass

    def vertex_index_list(self):
        return self.vertices, self.indices


class Plane(SceneObject):
    def __init__(self, pos, normal, material=None):
        super(Plane, self).__init__(pos, material)
        self.normal = normal

    def intersection(self, ray):
        t = np.dot(ray.origin, self.normal) / np.dot(self.normal, ray.direction)
        return ray.point(t)

    @property
    def vertices(self):
        return np.array([[-1.0, -1.0, 0.0],
                         [1.0, -1.0, 0.0],
                         [1.0, 1.0, 0.0],
                         [-1.0, 1.0, 0.0]])

    @property
    def indices(self):
        return np.array([[0, 1, 2], [0, 2, 3]])


class Disk(Plane):
    def __init__(self, pos, normal, radius, material=None):
        super(Disk, self).__init__(pos, normal, material)
        self.radius = radius

    def is_inside(self, pt):
        return np.sqrt(np.sum((pt - self.pos) ** 2)) <= self.radius

    def intersection(self, ray):
        pt = super(Disk, self).intersection(ray)
        if self.is_inside(pt):
            return pt
        return np.inf


class Sphere(SceneObject):
    def __init__(self, center, radius):
        super(Sphere, self).__init__(center)
        self.radius = radius

    @property
    def center(self):
        return self.pos

    def intersection(self, ray):
        """
        :param ray: [N x 3] rays (Considering N x 3 instead of 3 x N is so that we can left multiply, data flowing from
        right to left]
        :return:
        """

        """
        vectorized ray-sphere intersection for usage outside of sphere
        
        renderer.setup() # this will create vertex buffer object like data-structure

        centers = [c for c in Sphere.center]
        centers matrix, C = [3, M]
        ray matrix R = [N, 3]
        a = np.sum(R ** 2, axis=-1) == 1
        b = 2 * R * C
        c = np.sum(C, axis=0) - radius ** 2
        b +/- sqrt(b**2 - 4 * a * c) / (2 * a ** 2)

        b +/- sqrt(b ** 2 - 4 * c)

        """
        assert isinstance(ray, Ray)
        dir = ray.direction
        a = np.sum(dir ** 2, axis=-1)
        b = -2 * np.dot(dir, self.pos)
        c = np.sum(self.pos ** 2) - self.radius ** 2

        d = np.sqrt(b ** 2 - 4 * a * c)
        inv_denom = 1. / (2 * a)

        left_intersect = (-b - d) * inv_denom
        right_intersect = (-b + d) * inv_denom

        if left_intersect < 0:
            return right_intersect

        return left_intersect


