import numpy as np
import time


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


def uniform_sample_sphere(radius, num_samples, theta_range=None, phi_range=None, axis=None, angle=None):
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
