"""Sample generator."""
from diffrend.model import compute_face_normal
import diffrend.numpy.ops as ops
import numpy as np


def uniform_sample_circle(radius, num_samples, normal=np.array([0., 0., 1.])):
    """Generate uniform random samples into a circle."""
    theta = np.random.rand(num_samples) * 2 * np.pi
    return radius * np.stack((np.cos(theta), np.sin(theta),
                              np.zeros_like(theta)), axis=1)


def uniform_sample_cylinder(radius, height, num_samples,
                            normal=np.array([0., 0., 1.])):
    """Generate uniform random samples into a cilinder."""
    theta = np.random.rand(num_samples) * 2 * np.pi
    z = height * (np.random.rand(num_samples) - .5)
    return radius * np.stack((np.cos(theta), np.sin(theta), z), axis=1)


def uniform_sample_sphere_patch(radius, num_samples, theta_range, phi_range):
    """Generate uniform random samples a patch defined by theta and phi ranges
       on the surface of the sphere.
       :param theta_range: angle from the z-axis
       :param phi_range: range of angles on the xy plane from the x-axis
    """
    pts_2d = np.random.rand(num_samples, 2)
    s_range = 1 - np.cos(np.array(theta_range) / 2) ** 2
    t_range = np.array(phi_range) / (2 * np.pi)
    s = min(s_range) + pts_2d[:, 0] * (max(s_range) - min(s_range))
    t = min(t_range) + pts_2d[:, 1] * (max(t_range) - min(t_range))
    # theta is angle from the z-axis
    theta = 2 * np.arccos(np.sqrt(1 - s))
    phi = 2 * np.pi * t
    pts = np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                    np.cos(theta)), axis=1) * radius
    return pts


def uniform_sample_sphere_cone(radius, num_samples, axis, angle):
    """Generate uniform random samples a patch defined by theta and phi ranges
       on the surface of the sphere.
       :param theta_range: angle from the z-axis
       :param phi_range: range of angles on the xy plane from the x-axis
    """
    theta_range = [0, angle]
    phi_range = [0, 2 * np.pi]

    # Generate samples around the z-axis
    pts = uniform_sample_sphere_patch(radius, num_samples, theta_range=theta_range, phi_range=phi_range)

    # Transform from z-axis to the target axis
    axis = ops.normalize(axis)
    ortho_axis = np.cross([0, 0, 1], axis)
    ortho_axis_norm = ops.norm(ortho_axis)
    rot_angle = np.arccos(axis[2])
    if rot_angle > 0 and ortho_axis_norm > 0:
        pts_rot = ops.rotate_axis_angle(ortho_axis, rot_angle, pts)
    elif np.abs(rot_angle - np.pi) < 1e-12:
        pts_rot = pts
        pts_rot[..., 2] *= -1
    else:
        pts_rot = pts
    return pts_rot


def uniform_sample_full_sphere(radius, num_samples):
    """Generate uniform random samples into a sphere."""
    return uniform_sample_sphere_patch(radius, num_samples, theta_range=[0, np.pi],
                                       phi_range=[0, 2 * np.pi])


def uniform_sample_sphere(radius, num_samples, axis=None, angle=None, theta_range=None, phi_range=None):
    dispatch_table = [uniform_sample_full_sphere,
                      lambda x, y: uniform_sample_sphere_cone(x, y, axis=axis, angle=angle),
                      lambda x, y: uniform_sample_sphere_patch(x, y, theta_range=theta_range, phi_range=phi_range)]
    if axis is not None and angle is not None:
        assert theta_range is None and phi_range is None
        ver = 1
    elif theta_range is not None and phi_range is not None:
        assert axis is None and angle is None
        ver = 2
    else:
        assert axis is None and angle is None and theta_range is None and phi_range is None
        ver = 0

    return dispatch_table[ver](radius, num_samples)


# TODO: Unfinished
def uniform_sample_torus(inner_radius, outer_radius, num_samples,
                         normal=np.array([0., 0., 1.])):
    """Rejection sampling based method.

    From: https://math.stackexchange.com/questions/2017079/uniform-random-points-on-a-torus
    Here I use a different one that works in parallel. Need to check if this is
    correct.
    1. First generate samples between inner_radius and outer_radius based on
    probability weighted by [inner_rad, outer_rad]:
        shift to (outer+inner) / 2
    2. uniform randomly choose the sign of z and compute:
        rad = outer - inner
        r * cos(theta) = y => theta = arccos(x/r)
        z = r * sin(theta)
    3. uniformly choose phi, and rotate all the points by R(phi, z)
    """
    r = outer_radius - inner_radius
    R = (inner_radius + outer_radius) / 2.


def uniform_sample_triangle(v, num_samples):
    """Generate uniform random samples into a triangle."""
    samples = np.random.rand(num_samples, 2)
    # surface parameters
    s, t = samples[:, 0], samples[:, 1]
    # barycentric coordinates
    sqrt_s = np.sqrt(s)
    b = np.stack((1 - sqrt_s, (1 - t) * sqrt_s, t * sqrt_s), axis=1)

    if np.ndim(v) == 2:  # single triangle
        assert v.shape[0] == 3 and v.shape[1] == 3
        v = v[np.newaxis, ...]

    # first axis is number of faces
    assert np.ndim(v) == 3 and v.shape[1] == 3 and v.shape[2] == 3

    return np.squeeze(np.sum(b[..., np.newaxis] * v, axis=1))


def triangle_double_area(obj):
    """Triangle double area.

    https://github.com/alecjacobson/gptoolbox/blob/master/mesh/doublearea.m
    :param obj:
    :return:
    """
    v = obj['v']
    f = obj['f']

    v1, v2, v3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    if v.shape[-1] == 2:
        r = v1 - v3
        s = v2 - v3
        dblA = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    elif v.shape[-1] == 3:
        dblA = np.sqrt(triangle_double_area({'v': v[:, [1, 2]], 'f': f}) ** 2 +
                       triangle_double_area({'v': v[:, [2, 0]], 'f': f}) ** 2 +
                       triangle_double_area({'v': v[:, [0, 1]], 'f': f}) ** 2)
    else:
        raise ValueError("Not Implemented")

    return dblA


def uniform_sample_mesh(obj, num_samples, camera=None):
    """Generate uniform random samples on a mesh.
    :param obj: Dictionary with vertices in 'v' and faces 'f'
    :param num_samples: The number of samples to generate
    :param camera: If specified then generate only sample that are visible to the camera.
                   Camera is specified by {'eye': np.array([x, y, z]),
                                           'up': np.array([x, y, z]),
                                           'at': np.array([x, y, z])}
    :return: num_samples x 3 matrix, vertex normal
    """
    if camera is not None:
        obj = ops.backface_culling(obj, camera=camera, copy=True)
        # TODO: occlusion culling, frustum culling
        # ...

    v = obj['v']
    f = obj['f']
    if 'a' in obj:
        area = obj['a']
    else:
        area = triangle_double_area(obj)

    if 'fn' in obj:
        fn = obj['fn']
    else:
        fn = compute_face_normal(obj)

    prob_area = area / np.sum(area)

    # First choose triangles based on their size
    idx = np.random.choice(f.shape[0], num_samples, p=prob_area)
    # Construct batch of triangles
    tri = np.concatenate((v[f[idx, 0]][:, np.newaxis, :],
                          v[f[idx, 1]][:, np.newaxis, :],
                          v[f[idx, 2]][:, np.newaxis]),
                         axis=1)
    vn = fn[idx]
    return uniform_sample_triangle(tri, num_samples), vn


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()

    pts = uniform_sample_sphere(radius=1.0, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])

    pts = uniform_sample_sphere(radius=1.0, num_samples=1000, axis=None, theta_range=[0, np.pi/2],
                                phi_range=[0, np.pi/2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    pts = uniform_sample_sphere(radius=1.0, num_samples=1000, axis=None, theta_range=[0, np.pi / 2],
                                phi_range=[np.pi / 6, np.pi / 3])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    pts = uniform_sample_sphere(radius=2.0, num_samples=1000, axis=None, theta_range=[np.pi / 4, np.pi / 3],
                                phi_range=[np.pi / 6, np.pi / 3])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    pts = uniform_sample_sphere(radius=1.0, num_samples=1000, axis=None, theta_range=[0, np.pi / 2],
                                phi_range=[0, 2 * np.pi])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    pts = uniform_sample_sphere(radius=1.0, num_samples=1000, axis=None, theta_range=[np.pi / 2, np.pi / 2],
                                phi_range=[0, 2 * np.pi])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    pts = uniform_sample_sphere(radius=1.0, num_samples=100, axis=None, theta_range=[0, np.pi / 8],
                                phi_range=[0, 2 * np.pi])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('Pole')

    pts = uniform_sample_sphere(radius=1.0, num_samples=100, axis=[1, 1, 1], angle=np.pi / 36)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('Cone')

    pts = uniform_sample_sphere(radius=1.0, num_samples=100, axis=[0, 0, -0.99], angle=np.pi / 36)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('-Z Cone')

    pts = uniform_sample_circle(radius=1.0, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])

    pts = uniform_sample_cylinder(radius=0.25, height=1.0, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])

    v = np.array([[0., 0., 0.], [1., 0., 0.], [0.5, 1.0, 0.]])
    pts = uniform_sample_triangle(v, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], 'r.')

    v = np.array([[0., 0., 0.], [1., 0., 0.], [0.5, 1.0, 0.], [2., 2., 1.]])
    f = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    pts, vn = uniform_sample_mesh({'v': v, 'f': f}, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')

    from diffrend.model import load_model

    obj = load_model('../../data/chair_0001.off')
    pts, vn = uniform_sample_mesh(obj, num_samples=1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.6)
    plt.xlabel('x')
    plt.ylabel('y')

    obj = load_model('../../data/bunny.obj')
    pts_obj, vn = uniform_sample_mesh(obj, num_samples=800)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_obj[:, 0], pts_obj[:, 1], pts_obj[:, 2])
    ax.view_init(93, -64)
    plt.xlabel('x')
    plt.ylabel('y')

    obj = load_model('../../data/desk_0007.off')
    camera = {'eye': np.array([0, 0, 10])}
    pts_obj, vn = uniform_sample_mesh(obj, num_samples=1000, camera=camera)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_obj[:, 0], pts_obj[:, 1], pts_obj[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')

    obj = load_model('../../data/bunny.obj')
    camera = {'eye': np.array([0, 0, 10])}
    pts_obj, vn = uniform_sample_mesh(obj, num_samples=800, camera=camera)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_obj[:, 0], pts_obj[:, 1], pts_obj[:, 2])
    ax.view_init(93, -64)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
