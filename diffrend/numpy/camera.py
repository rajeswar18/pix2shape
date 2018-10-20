import numpy as np
from diffrend.numpy.vector import Vector
from diffrend.numpy.quaternion import Quaternion
import diffrend.numpy.ops as ops


class Camera(object):
    def __init__(self, pos, at, up, viewport):
        #assert isinstance(orientation, Quaternion)
        self.pos = np.array(pos, dtype=np.float32)

        if self.pos.size == 3:
            self.pos = np.append(self.pos, 1.0)

        #self.orientation = orientation
        self.at = np.array(at, dtype=np.float32)

        if self.at.size == 3:
            self.at = np.append(self.eye, 1.0)

        self.up = ops.normalize(np.array(up))
        self.viewport = viewport
        self.view_matrix = ops.lookat(eye=self.pos, at=self.at, up=self.up)

    def __str__(self):
        #return 'Camera: pos {}, orientation: {}'.format(self.pos, self.orientation)
        return 'Camera: pos {}, at: {}, up: {}'.format(self.pos, self.at, self.up)

    @property
    def eye(self):
        return self.pos

    @property
    def aspect_ratio(self):
        return self.viewport[2] / self.viewport[3]

    @property
    def M(self):
        return self.view_matrix

    def rotate(self, axis, angle):
        self.orientation = self.orientation.rotate(angle_rad=angle, axis=axis)

    def translate(self, translation):
        self.pos[:3] += translation[:3]

    def lookat(self, eye, at, up):
        """Same as the global lookat but changes the state of the current camera
        :param eye:
        :param at:
        :param up:
        :return:
        """
        if type(eye) is list:
            eye = np.array(eye, dtype=np.float32)
        if eye.size == 3:
            eye = np.append(eye, 1.0)
        self.pos = eye
        self.view_matrix = ops.lookat(self.pos, at, up)

    def generate_rays(self):
        pass


class PinholeCamera(Camera):
    def __init__(self, pos, at, up, fovy, focal_length, viewport):
        """
        :param pos:
        :param at:
        :param up:
        :param fovy: Vertical field of view in radians
        :param focal_length:
        :param viewport:
        """
        super(PinholeCamera, self).__init__(pos, at, up, viewport)
        self.fovy = float(fovy)
        self.focal_length = float(focal_length)
        self.viewport = viewport
        height = 2 * self.focal_length * np.tan(self.fovy / 2.)
        aspect_ratio = float(viewport[2]) / viewport[3]
        width = height * aspect_ratio

        self.proj_matrix = np.array([[self.focal_length, 0, 0, 0],
                                    [0, self.focal_length, 0, 0],
                                    [0, 0, self.focal_length, 0],
                                    [0, 0, 1, 0]])

    @property
    def M(self):
        return np.dot(self.proj_matrix, self.model_view)

    @property
    def viewport_matrix(self):
        return None

    @property
    def projection(self):
        return self.proj_matrix

    @property
    def model_view(self):
        w = 1
        if self.pos.size == 4:
            w = self.pos[3]
        translation_matrix = np.array([[1, 0, 0, -self.pos[0] / w],
                                       [0, 1, 0, -self.pos[1] / w],
                                       [0, 0, 1, -self.pos[2] / w],
                                       [0, 0, 0, 1]])
        return np.dot(translation_matrix, self.orientation.R, )


class TrackBallCamera(PinholeCamera):
    def __init__(self, pos, up, fovy, focal_length, viewport):
        super(TrackBallCamera, self).__init__(pos, np.array([0., 0., 0., 1.]), up, fovy, focal_length, viewport)
        self.model_matrix = np.eye(4)

    def screen_to_sphere(self, coords):
        w, h = self.viewport[2] - self.viewport[0], self.viewport[3] - self.viewport[1]
        cx, cy = w/2, h/2
        radius = min(cx, cy)
        #print(cx, cy, radius)
        x = (coords[0] - cx) / radius
        y = -(coords[1] - cy) / radius
        r_sqr = x ** 2 + y ** 2
        if r_sqr > 1:
            s = 1 / np.sqrt(r_sqr)
            x *= s
            y *= s
            z = 0
        else:
            z = np.sqrt(1 - r_sqr)
        return np.array([x, y, z])

    def mouse_press(self, coords):
        self.src = self.screen_to_sphere(coords)

    def mouse_move(self, coords):
        self.dst = self.screen_to_sphere(coords)
        #print('src', self.src, 'dst:', self.dst)
        # compute object rotation
        axis = ops.normalize(np.cross(self.src, self.dst))
        theta = np.arccos(np.dot(self.src, self.dst))
        #print('axis of rotation ', axis)
        #print('rotation amount ', theta)
        #self.rotate(axis=axis, angle=theta)
        # cam_pos = ops.rotate_axis_angle(axis=axis, angle=theta, vec=self.pos)
        # cam_up = ops.rotate_axis_angle(axis=axis, angle=theta, vec=self.up)
        # print('cam_pos ', cam_pos)
        # print('cam_up ', cam_up)
        self.model_matrix = np.matmul(self.model_matrix, ops.axis_angle_matrix(axis=axis, angle=theta))
        #self.pos = cam_pos
        #self.up = cam_up

        self.src = self.dst
        #self.view_matrix = self.orientation.R

    def zoom(self, amount):
        delta = ops.normalize(self.pos) * amount
        self.translate(delta)


class VirtualSphereCamera(PinholeCamera):
    def __init__(self, pos, up, fovy, focal_length, viewport):
        from diffrend.numpy.geometry import Sphere
        super(VirtualSphereCamera, self).__init__(pos, np.array([0., 0., 0., 1.]), up, fovy, focal_length, viewport)
        self.cam_dir = ops.normalize(self.pos[:3])
        self.radius, self.phi, self.theta = ops.cart2sph(self.cam_dir[0], self.cam_dir[1], self.cam_dir[2])
        self.sphere = Sphere(center=np.array([0., 0., 0.]), radius=self.radius)

    def mouse_press(self, coords):
        self.src = ops.normalize([coords[0], coords[1], self.pos[2] - self.focal_length])
        # pick point on sphere
        # ...

    def mouse_move(self, coords):
        self.dst = ops.normalize([coords[0], coords[1], self.pos[2] - self.focal_length])
        delta_step = self.dst - self.src

        if ops.norm(delta_step) == 0:
            return

        print('src', self.src, 'dst:', self.dst)

        # compute object rotation
        axis = ops.normalize(np.cross(self.src, self.dst))
        theta = np.arccos(np.dot(self.src, self.dst))
        print(axis, theta)
        self.cam_dir = ops.rotate_axis_angle(axis=axis, angle=theta, vec=self.cam_dir)
        print('cam_dir', self.cam_dir)

        self.phi += delta_step[0] * 0.1
        self.theta += delta_step[1] * 0.1
        print('delta ', delta_step)
        cam_pos = ops.sph2cart(radius=self.radius, phi=self.phi, theta=self.theta)
        print('cam_pos', cam_pos)
        #self.pos = cam_pos
        # self.rotate(axis=axis, angle=theta)
        #
        self.src = self.dst
        # self.view_matrix = self.orientation.R

    def zoom(self, amount):
        delta = ops.normalize(self.pos) * amount
        self.translate(delta)


if __name__ == '__main__':
    cam = VirtualSphereCamera([0., 0., 1., 1.], up=[0, 1, 0], fovy=45, focal_length=1.,
                              viewport=[0, 0, 640, 480])

    cam.mouse_press([0, 0])
    cam.mouse_move([1, 0])
    print(cam)

    cam = TrackBallCamera([0.0, 0.0, 1.0, 1.0], up=[0, 1, 0], fovy=45, focal_length=0.01,
                          viewport=[0, 0, 640, 480])
    cam.mouse_press([640/2, 480/2])
    cam.mouse_move([640/2 + 2, 480/2])
