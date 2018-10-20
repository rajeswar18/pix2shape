import numpy as np
import torch
from torch.autograd import Variable


# TODO: SPLIT THIS INTO UTILS AND OPS (like diffrend.numpy)
CPU_ONLY = False
if torch.cuda.is_available() and not CPU_ONLY:
    CUDA = True
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    CUDA = False
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

print('CUDA support ', CUDA)

tch_var = lambda x, fn_type, req_grad: Variable(fn_type(x),
                                                requires_grad=req_grad)
tch_var_f = lambda x: tch_var(x, FloatTensor, False)
tch_var_l = lambda x: tch_var(x, LongTensor, False)


def np_var(x, req_grad=False):
    """Convert a numpy variable to a pytorch variable."""
    if CUDA:
        return Variable(torch.from_numpy(x), requires_grad=req_grad).cuda()
    else:
        return Variable(torch.from_numpy(x), requires_grad=req_grad)

# np_var_f = lambda x: np_var(x, FloatTensor, False)
# np_var_l = lambda x: np_var(x, LongTensor, False)


def get_data(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def bincount(x, nbins, ret_var=True):
    data = x.cpu().data if x.is_cuda else x.data
    count = torch.zeros(nbins).scatter_add_(0, data, torch.ones(x.size()))
    if ret_var:
        count = torch.autograd.Variable(count)
    return count


def where(cond, x, y):
    return cond.float() * x + (1 - cond.float()) * y


def norm2_sqr(u):
    return torch.sum(u ** 2, dim=-1)


def norm_p(u, p=2, eps=0):
    return torch.pow(torch.sum(torch.pow(u, p) + eps, dim=-1), 1./p)


def tensor_dot(x, y, axis=0):
    return torch.sum(x * y, dim=axis)


def tensor_cross_prod(u, M):
    """
    :param u:  N x 3
    :param M: N x P x 3
    :return:
    """
    s0 = u[:, 1][:, np.newaxis] * M[..., 2] - u[:, 2][:, np.newaxis] * M[..., 1]
    s1 = -u[:, 0][:, np.newaxis] * M[..., 2] + u[:, 2][:, np.newaxis] * M[..., 0]
    s2 = u[:, 0][:, np.newaxis] * M[..., 1] - u[:, 1][:, np.newaxis] * M[..., 0]

    return torch.stack((s0, s1, s2), dim=2)

def nonzero_divide(x, y):
    """ x and y need to have the same dimensions.
    :param x:
    :param y:
    :return:
    """
    assert list(x.size()) == list(y.size())

    mask = torch.abs(y) > 0
    return x.masked_scatter(mask, x.masked_select(mask) / y.masked_select(mask))


def unit_norm2_L2loss(x, scale):
    """
    :param x: [N, 3] matrix
    :return: Scalar loss
    """
    return torch.mean((scale * (torch.sqrt(norm2_sqr(x)) - 1)) ** 2)


def unit_norm2_L1loss(x, scale):
    """
    :param x: [N, 3] matrix
    :return: Scalar loss
    """
    return torch.mean(scale * torch.abs(torch.sqrt(norm2_sqr(x)) - 1))


def unit_norm2sqr_L2loss(x, scale):
    """
    :param x: [N, 3] matrix
    :return: Scalar loss
    """
    return torch.mean((scale * (norm2_sqr(x) - 1)) ** 2)


def unit_norm2sqr_L1loss(x, scale):
    """
    :param x: [N, 3] matrix
    :return: Scalar loss
    """
    return torch.mean(scale * torch.abs(norm2_sqr(x) - 1))


def normalize_maxmin(x):
    min_val = torch.min(x)
    return (x - min_val) / (torch.max(x) - min_val)


def normalize(u, eps=1e-10):
    denom = norm_p(u, 2, eps=eps)
    if u.dim() > 1:
        denom = denom[..., np.newaxis]
    # TODO: nonzero_divide for rows with norm = 0
    return u / (denom + eps)
    # if u.dim() == 2:
    #     return torch.renorm(u, 2, 0, 1)
    # elif u.dim() == 3:
    #     return torch.renorm(u, 2, )



def reflect_ray(incident, normal):
    """
    :param incident: L x N x 3 matrix
    :param normal: 1 x N x 3 matrix
    :return: L x N x 3 matrix
    """
    return -2 * torch.sum(incident * normal, dim=-1)[..., np.newaxis] * normal + incident


def point_along_ray(ray_orig, ray_dir, ray_dist):
    """Find the point along the ray_dir at distance ray_dist
    :param ray_orig: 4-element vector or 3-element vector or [N x 3] matrix
    :param ray_dir: [4 x N] matrix with N rays and each ray being [x, y, z, 0] direction or [3 x N]
    :param ray_dist: [M x N] matrix with M objects and N rays
    :return: [M x N x 4] intersection points or [M x N x 3]
    """
    #return ray_orig[np.newaxis, np.newaxis, :] + ray_dist[:, :, np.newaxis] * ray_dir.transpose(1, 0)[np.newaxis, ...]
    return ray_orig[np.newaxis, ...] + ray_dist[:, :, np.newaxis] * ray_dir.transpose(1, 0)[np.newaxis, ...]


def ray_sphere_intersection(ray_orig, ray_dir, sphere, **kwargs):
    """Bundle of rays intersected with a batch of spheres
    :param eye:
    :param ray_dir:
    :param sphere:
    :return:
    """
    pos = sphere['pos'][:, :3]
    pos_tilde = ray_orig[np.newaxis, ...] - pos[:, np.newaxis, :]
    #pos_tilde = ray_orig - pos
    radius = sphere['radius']

    a = torch.sum(ray_dir ** 2, dim=0)
    b = 2 * torch.sum(pos_tilde * ray_dir.permute(1, 0)[np.newaxis, ...], dim=-1)
    c = (torch.sum(pos_tilde ** 2, dim=-1) - radius[:, np.newaxis] ** 2)

    d_sqr = b ** 2 - 4 * a * c
    intersection_mask = d_sqr >= 0

    d_sqr = where(intersection_mask, d_sqr, 0)

    d = torch.sqrt(d_sqr)
    inv_denom = 1. / (2 * a)

    t1 = (-b - d) * inv_denom
    t2 = (-b + d) * inv_denom

    # get the nearest positive depth
    max_val = torch.max(torch.max(t1, t2)) + 1
    t1 = where(intersection_mask * (t1 >= 0), t1, max_val)
    t2 = where(intersection_mask * (t2 >= 0), t2, max_val)

    ray_dist, _ = torch.min(torch.stack((t1, t2), dim=2), dim=2)
    ray_dist = where(intersection_mask, ray_dist, 1001)

    intersection_pts = point_along_ray(ray_orig, ray_dir, ray_dist)

    normals = normalize(intersection_pts - pos[:, np.newaxis, :])

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_plane_intersection(ray_orig, ray_dir, plane, **kwargs):
    """Intersection a bundle of rays with a batch of planes
    :param eye: Camera's center of projection
    :param ray_dir: Ray direction
    :param plane: Plane specification
    :return:
    """
    pos = plane['pos'][:, :3]
    normal = normalize(plane['normal'][:, :3])
    dist = torch.sum(pos * normal, dim=1)

    denom = torch.mm(normal, ray_dir)

    # check for denom = 0
    intersection_mask = torch.abs(denom) > 0

    ray_dist = (dist.unsqueeze(-1) - torch.mm(normal, ray_orig.permute(1, 0))) / denom

    intersection_pts = point_along_ray(ray_orig, ray_dir, ray_dist)

    if 'disable_normals' in kwargs and kwargs['disable_normals']:
        normals = None
    else:
        # TODO: Delay this normal selection to save memory. Because even if there's an intersection it may not be visible
        normals = normal[:, np.newaxis, :].repeat(1, intersection_pts.size()[1], 1)

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_disk_intersection(ray_orig, ray_dir, disks, **kwargs):
    result = ray_plane_intersection(ray_orig, ray_dir, disks, **kwargs)
    intersection_pts = result['intersect']
    normals = result['normal']
    ray_dist = result['ray_distance']

    centers = disks['pos'][:, :3]
    radius = disks['radius']
    dist_sqr = torch.sum((intersection_pts - centers[:, np.newaxis, :]) ** 2, dim=-1)

    # Intersection mask
    intersection_mask = (dist_sqr <= radius[:, np.newaxis] ** 2)
    ray_dist = where(intersection_mask, ray_dist, 1001)

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_triangle_intersection(ray_orig, ray_dir, triangles, **kwargs):
    """Intersection of a bundle of rays with a batch of triangles.
    Assumes that the triangles vertices are specified as F x 3 x 4 matrix where F is the number of faces and
    the normals for all faces are precomputed and in a matrix of size F x 4 (i.e., similar to the normals for other
    geometric primitives). Note that here the number of faces F is the same as number of primitives M.
    :param ray_orig:
    :param ray_dir:
    :param triangles:
    :return:
    """

    planes = {'pos': triangles['face'][:, 0, :], 'normal': triangles['normal']}
    result = ray_plane_intersection(ray_orig, ray_dir, planes)
    intersection_pts = result['intersect']  # M x N x 4 matrix where M is the number of objects and N pixels.
    normals = result['normal'][..., :3]  # M x N x 4
    ray_dist = result['ray_distance']

    # check if intersection point is inside or outside the triangle
    # M x N x 3
    v_p0 = (intersection_pts - triangles['face'][:, 0, :3][:, np.newaxis, :])
    v_p1 = (intersection_pts - triangles['face'][:, 1, :3][:, np.newaxis, :])
    v_p2 = (intersection_pts - triangles['face'][:, 2, :3][:, np.newaxis, :])

    # Torch and Tensorflow's cross product requires both inputs to be of the same size unlike numpy
    # M x 3
    v01 = triangles['face'][:, 1, :3] - triangles['face'][:, 0, :3]
    v12 = triangles['face'][:, 2, :3] - triangles['face'][:, 1, :3]
    v20 = triangles['face'][:, 0, :3] - triangles['face'][:, 2, :3]

    cond_v01 = torch.sum(tensor_cross_prod(v01, v_p0) * normals, dim=-1) >= 0
    cond_v12 = torch.sum(tensor_cross_prod(v12, v_p1) * normals, dim=-1) >= 0
    cond_v20 = torch.sum(tensor_cross_prod(v20, v_p2) * normals, dim=-1) >= 0

    intersection_mask = cond_v01 * cond_v12 * cond_v20
    ray_dist = where(intersection_mask, ray_dist, 1001)

    return {'intersect': intersection_pts, 'normal': result['normal'], 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


intersection_fn = {'disk': ray_disk_intersection,
                   'plane': ray_plane_intersection,
                   'sphere': ray_sphere_intersection,
                   'triangle': ray_triangle_intersection,
                   }


def lookat(eye, at, up):
    """Returns a lookat matrix
    :param eye:
    :param at:
    :param up:
    :return:
    """
    return lookat_inv(eye, at, up).inverse()


def lookat_inv(eye, at, up):
    """Returns the inverse lookat matrix
    :param eye: camera location
    :param at: lookat point
    :param up: up direction
    :return: 4x4 inverse lookat matrix
    """
    rot_matrix = lookat_rot_inv(eye, at, up)
    rot_translate = torch.cat((rot_matrix, eye[:3][:, np.newaxis]), dim=1)
    return torch.cat((rot_translate, tch_var_f([0, 0, 0, 1.])[np.newaxis, :]), dim=0)



def lookat_rot_inv(eye, at, up):
    """Returns the inverse lookat matrix
    :param eye: camera location
    :param at: lookat point
    :param up: up direction
    :return: 4x4 inverse lookat matrix
    """
    if up.size()[-1] == 4:
        assert get_data(up)[3] == 0
        up = up[:3]

    if eye.size()[-1] == 4:
        assert abs(get_data(eye)[3]) > 0
        eye = eye[:3] / eye[3]

    if at.size()[-1] == 4:
        assert abs(get_data(at)[3]) > 0
        at = at[:3] / at[3]

    z = normalize(eye - at)
    up = normalize(up)
    x = normalize(torch.cross(up, z))
    # The input `up` vector may not be orthogonal to z.
    y = torch.cross(z, x)

    return torch.stack((x, y, z), dim=1)


def tonemap(im, **kwargs):
    if kwargs['type'] == 'gamma':
        return torch.pow(im, kwargs['gamma'])


def generate_rays(camera):
    viewport = np.array(camera['viewport'])
    W, H = viewport[2] - viewport[0], viewport[3] - viewport[1]
    aspect_ratio = W / H

    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
    n_pixels = x.size

    fovy = np.array(camera['fovy'])
    focal_length = np.array(camera['focal_length'])
    h = np.tan(fovy / 2) * 2 * focal_length
    w = h * aspect_ratio

    x *= w / 2
    y *= h / 2

    x = tch_var_f(x.ravel())
    y = tch_var_f(y.ravel())

    eye = camera['eye'][:3]
    at = camera['at'][:3]
    up = camera['up'][:3]

    proj_type = camera['proj_type']
    if proj_type == 'ortho' or proj_type == 'orthographic':
        ray_dir = normalize(at - eye)[:, np.newaxis]
        ray_orig = torch.stack((x, y, tch_var_f(np.zeros(n_pixels)), tch_var_f(np.ones(n_pixels))), dim=0)
        inv_view_matrix = lookat_inv(eye=eye, at=at, up=up)
        ray_orig = torch.mm(inv_view_matrix, ray_orig)
        ray_orig = (ray_orig[:3] / ray_orig[3][np.newaxis, :]).permute(1, 0)
    elif proj_type == 'persp' or proj_type == 'perspective':
        ray_orig = eye[np.newaxis, :]
        ray_dir = torch.stack((x, y, tch_var_f(-np.ones(n_pixels) * focal_length)), dim=0)
        inv_view_matrix = lookat_rot_inv(eye=eye, at=at, up=up)
        ray_dir = torch.mm(inv_view_matrix, ray_dir)

        # normalize ray direction
        ray_dir /= torch.sqrt(torch.sum(ray_dir ** 2, dim=0))

    return ray_orig, ray_dir, H, W


def ray_object_intersections(eye, ray_dir, scene_objects, **kwargs):
    obj_intersections = None
    ray_dist = None
    normals = None
    material_idx = None
    for obj_type in scene_objects:
        result = intersection_fn[obj_type](eye, ray_dir, scene_objects[obj_type], **kwargs)
        curr_intersects = result['intersect']
        curr_ray_dist = result['ray_distance']
        curr_normals = result['normal']
        if curr_ray_dist.dim() == 1:
            curr_ray_dist = curr_ray_dist[np.newaxis, :]
        if curr_intersects.dim() == 1:
            curr_intersects = curr_intersects[np.newaxis, :]
        if curr_normals is not None and curr_normals.dim() == 1:
            curr_normals = curr_normals[np.newaxis, :]

        if obj_intersections is None:
            assert ray_dist is None
            obj_intersections = curr_intersects
            ray_dist = curr_ray_dist
            normals = curr_normals
            material_idx = scene_objects[obj_type]['material_idx']
        else:
            obj_intersections = torch.cat((obj_intersections, curr_intersects), dim=0)
            ray_dist = torch.cat((ray_dist, curr_ray_dist), dim=0)
            if normals is not None:
                normals = torch.cat((normals, curr_normals), dim=0)
            material_idx = torch.cat((material_idx, scene_objects[obj_type]['material_idx']), dim=0)

    return obj_intersections, ray_dist, normals, material_idx


def backface_labeler(eye, scene_objects):
    """Add a binary label per planar geometry.
       0: Facing the camera.
       1: Facing away from the camera, i.e., back-face.
    :param eye: Camera position
    :param scene_objects: Dictionary of scene geometry
    :return: Dictionary of scene geometry with backface label for each geometry
    """
    for obj_type in scene_objects:
        if obj_type == 'sphere':
            continue
        if obj_type == 'triangle':
            pos = scene_objects[obj_type]['face'][:, 0, :3]
        else:
            pos = scene_objects[obj_type]['pos'][:, :3]
        normals = scene_objects[obj_type]['normal'][:, :3]
        cam_dir = normalize(eye[:3] - pos)
        facing_dir = torch.sum(normals * cam_dir, dim=-1)
        scene_objects[obj_type]['facing_dir'] = facing_dir
        scene_objects[obj_type]['backface'] = facing_dir < 0

    return scene_objects


def world_to_cam(pos, normal, camera):
    """Transforms from the camera coordinate to the world coordinate
    :param pos_normal: Assumes N x 3 or N x 4 position and normals
    :param camera: Camera specification. Only eye, at, and up are needed
    :return: positions and normals in the world coordinate. N x 3
    """
    eye = camera['eye'][:3]
    at = camera['at'][:3]
    up = camera['up'][:3]

    pos_CC = None
    normal_CC = None

    if pos is not None:
        view_matrix = lookat(eye=eye, at=at, up=up)
        if pos.size()[1] == 3:
            pos = torch.cat((pos, tch_var_f(np.ones(pos.size()[0])[:, np.newaxis])), dim=1)
        pos_CC = torch.mm(pos, view_matrix.transpose(1, 0))

    if normal is not None:
        inv_view_matrix = lookat_inv(eye=eye, at=at, up=up)
        normal_WC = normal[:, :3]
        # M^{-T}n = (n^T M^{-1})^T
        normal_CC = torch.mm(normal_WC, inv_view_matrix[:3, :3])

    return {'pos': pos_CC, 'normal': normal_CC}


def cam_to_world(pos, normal, camera):
    """Transforms from the camera coordinate to the world coordinate
    :param pos_normal: Assumes N x 3 or N x 4 position and normals
    :param camera: Camera specification. Only eye, at, and up are needed
    :return: positions and normals in the world coordinate. N x 3
    """
    eye = camera['eye'][:3]
    at = camera['at'][:3]
    up = camera['up'][:3]
    inv_view_matrix = lookat_inv(eye=eye, at=at, up=up)

    pos_WC = None
    normal_WC = None

    if pos is not None:
        if pos.size()[1] == 3:
            pos = torch.cat((pos, tch_var_f(np.ones(pos.size()[0])[:, np.newaxis])), dim=1)
        pos_WC = torch.mm(pos, inv_view_matrix.transpose(1, 0))

    if normal is not None:
        view_matrix = lookat(eye=eye, at=at, up=up)
        normal_CC = normal[:, :3]
        normal_WC = torch.mm(normal_CC, view_matrix[:3, :3])

    return {'pos': pos_WC, 'normal': normal_WC}


def test_cam_to_world_identity():
    """The camera is at the world origin"""
    camera = {'eye': tch_var_f([0, 0, 0, 1]),
              'at': tch_var_f([0, 0, -1, 1]),
              'up': tch_var_f([0, 1, 0, 0])
              }
    pos_CC = tch_var_f([[0, 0, 0, 1],
                        [0, 0, -1, 1],
                        ])
    normal_CC = tch_var_f([[0, 0, 1, 0],
                           [0, 1, 0, 0]],
                          )
    world_coord = cam_to_world(pos=pos_CC, normal=normal_CC, camera=camera)
    print(world_coord)
    pos_WC = get_data(world_coord['pos'])
    normal_WC = get_data(world_coord['normal'])
    np.testing.assert_equal(get_data(pos_CC[:, :3]), pos_WC[:, :3])
    np.testing.assert_equal(get_data(normal_CC[:, :3]), normal_WC[:, :3])


def test_cam_to_world_offset0():
    """The camera is at the world origin"""
    camera = {'eye': tch_var_f([0, 0, 1, 1]),
              'at': tch_var_f([0, 0, -1, 1]),
              'up': tch_var_f([0, 1, 0, 0])
              }
    pos_CC = tch_var_f([[0, 0, 0, 1],
                        [0, 0, -1, 1],
                        ])
    normal_CC = tch_var_f([[0, 0, 1, 0],
                           [0, 1, 0, 0]],
                          )
    pos_WC_gt = np.array([[0., 0., 1., 1.],
                        [0., 0., 0., 1.],
                        ])
    world_coord = cam_to_world(pos=pos_CC, normal=normal_CC, camera=camera)
    print(world_coord)
    pos_WC = get_data(world_coord['pos'])
    normal_WC = get_data(world_coord['normal'])
    np.testing.assert_equal(pos_WC_gt[:, :3], pos_WC[:, :3])
    np.testing.assert_equal(get_data(normal_CC[:, :3]), normal_WC[:, :3])


def test_cam_to_world_offset1():
    """The camera is at the world origin"""
    camera = {'eye': tch_var_f([0, 0, 1, 1]),
              'at': tch_var_f([0, 0, -1, 1]),
              'up': tch_var_f([0, 1, 0, 0])
              }
    pos_CC = tch_var_f([[0, 0, 0],
                        [0, 0, -1],
                        ])
    normal_CC = tch_var_f([[0, 0, 1, 0],
                           [0, 1, 0, 0]],
                          )
    pos_WC_gt = np.array([[0., 0., 1., 1.],
                        [0., 0., 0., 1.],
                        ])
    world_coord = cam_to_world(pos=pos_CC, normal=normal_CC, camera=camera)
    print(world_coord)
    pos_WC = get_data(world_coord['pos'])
    normal_WC = get_data(world_coord['normal'])
    np.testing.assert_equal(pos_WC_gt[:, :3], pos_WC[:, :3])
    np.testing.assert_equal(get_data(normal_CC[:, :3]), normal_WC[:, :3])


def test_tform_cc_wc():
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    fovy = np.deg2rad(45.)
    focal_length = 1.0
    h = np.tan(fovy / 2) * 2 * focal_length
    aspect_ratio = 1
    w = h * aspect_ratio

    x *= w / 2
    y *= h / 2

    x = tch_var_f(x.ravel())
    y = tch_var_f(y.ravel())


def away_from_camera_penalty(pos, normal, camera_pos=None):
    """
    Args:
        pos:
        normal:
        camera_pos: None if position are in the camera's coordinate system

    Returns:

    """
    if camera_pos is not None:
        cam_dir = normalize(camera_pos[np.newaxis, :3] - pos.view(-1, 3))
    else:
        cam_dir = normalize(-pos.view(-1, 3))
    return torch.sum(torch.nn.functional.relu(-tensor_dot(normal.view(-1, 3), cam_dir, axis=-1)))


def pad2d(x, pad, pad_type):
    FN_PADDING_TYPE_MAP = {'replicate': torch.nn.ReplicationPad2d,
                           'reflect': torch.nn.ReflectionPad2d,
                           }
    if x.dim() == 2:
        xx = x[np.newaxis, :, :, np.newaxis]
    elif x.dim() == 3:
        xx = x[np.newaxis, ...]
    elif x.dim() == 4:
        xx = x
    else:
        print(x.dim())
        raise ValueError('Unsupported number of dimensions.')

    xx = xx.transpose(3, 1)
    xx = FN_PADDING_TYPE_MAP[pad_type](pad)(xx).transpose(1, 3)

    if x.dim() == 2:
        xx = xx.squeeze()
    elif x.dim() == 3:
        xx = xx[0]
    return xx


def grad_spatial2d(x, pad_type='reflect'):
    """
    Args:
        x: 3D Tensor [H, W, C]

    Returns:
        4D Tensor [8, H, W, C]

    """
    x = pad2d(x, (1, 1, 1, 1), pad_type)
    H, W = x.shape[:2]
    center = x[1:-1, 1:-1, :]
    nbhr_diff = []
    delta = [-1, 0, 1]
    for dy in delta:
        for dx in delta:
            if dx == 0 and dy == 0:
                continue
            nbhr_diff.append(x[1 + dy:H + dy - 1, 1 + dx:W + dx - 1, :] - center)

    return torch.stack(nbhr_diff, dim=0)


def spatial_3x3(pos, norm=1):
    nbhr_diff = grad_spatial2d(pos)
    return torch.mean(torch.pow(torch.sum(torch.abs(nbhr_diff) ** norm, dim=-1), 1 / norm))


def depth_rgb_gradient_consistency(image, depth):
    """
    Args:
        image: RGB image
        depth: Depth image

    Returns: consistency loss. The gradients should be the same for diffusely lit scene

    """
    im_grad = grad_spatial2d(torch.mean(image, dim=-1)[..., np.newaxis])
    depth_grad = grad_spatial2d(depth[..., np.newaxis])
    return torch.mean(torch.abs(torch.abs(im_grad) - torch.abs(depth_grad)))


def normal_consistency_cost(pos, normal, norm):
    """
    Args:
        pos: [N, 3] position
        normal: [N, 3] normals

    Returns:
        mean of cosine difference from the true normal

    """
    # Unit vectors on a grid
    pos_vector = normalize(grad_spatial2d(pos), 1e-10)
    # Per-pixel cosine difference
    dot_prod = torch.sum(pos_vector * normal[np.newaxis, ...], dim=-1)
    # We want the |cosine_difference|^norm value to be exactly zero
    return torch.mean(torch.abs(dot_prod) ** norm)


def find_average_normal(pos, kernel_size):
    """Estimate the normal from the average normal of the local patches
    Args:
        pos:
        kernel_size:

    Returns:

    """
    nbhr_diff = normalize(grad_spatial2d(pos), 1e-10)
    # cross-prod of neighboring difference
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # In grad_spatial2d the middle difference is ignored
    # 0 1 2
    # 3 - 4
    # 5 6 7
    # Take cross products in counter-clockwise direction
    normal = torch.stack([torch.cross(nbhr_diff[4], nbhr_diff[2], dim=-1),
                          torch.cross(nbhr_diff[2], nbhr_diff[1], dim=-1),
                          torch.cross(nbhr_diff[1], nbhr_diff[0], dim=-1),
                          torch.cross(nbhr_diff[0], nbhr_diff[3], dim=-1),
                          torch.cross(nbhr_diff[3], nbhr_diff[5], dim=-1),
                          torch.cross(nbhr_diff[5], nbhr_diff[6], dim=-1),
                          torch.cross(nbhr_diff[6], nbhr_diff[7], dim=-1),
                          torch.cross(nbhr_diff[7], nbhr_diff[4], dim=-1)], dim=0)
    if np.any(np.isnan(get_data(normal))):
        assert not np.any(np.isnan(get_data(normal)))
    return torch.clamp(normalize(torch.mean(normal, dim=0), 1e-10), 0.0, 1.0)


def estimate_surface_normals_plane_fit(pos, kernel_size):
    """Performs constrained plane estimation with the requirements that
    the splat position has to be on the plane and the normal has to be on
    the positive hemisphere facing the camera (we also enforced the second
    constraint when we generated normals). The setup is that the camera is
    looking towards the -z axis (negative z) and the normals (nx, ny, nz)
    need to have nz > 0.
    So, for position (x0, y0, z0) and some neighbor (x, y, z),

    nx (x - x0) + ny (y - y0) + nz (z - z0) = 0

    Setting nz to be some positive constant c gives,
    nx (x - x0) + ny (y - y0) = -c (z - z0)

    The code solves for (nx, ny) and renormalizes to have unit length.

    Args:
        pos: [num_position, 3]
        kernel_size: scalar indicating the neighborhood size (not used)

    Returns:
        normals: [num_positions, 3]

    """
    nbhr_diff = normalize(grad_spatial2d(pos), 1e-10)
    nbhr_diff = nbhr_diff.view(nbhr_diff.shape[0], -1, 3)
    M = nbhr_diff[:, :, :2].transpose(1, 0)
    Mt = M.transpose(2, 1)
    MtM = Mt.matmul(M)
    ad_m_bc = MtM[:, 0, 0] * MtM[:, 1, 1] - MtM[:, 0, 1] * MtM[:, 1, 0]
    # [[a, b], [c, d]] --> [[d, -b], [-c, a]]
    MtM = MtM.index_select(1, tch_var_l([1, 0])).transpose(2, 1).index_select(1, tch_var_l([1, 0])) * \
          tch_var_f([[1, -1], [-1, 1]])[np.newaxis, ...]
    MtMinv = MtM / (ad_m_bc[:, np.newaxis, np.newaxis] + 1e-12)
    # (M^TM)^{-1}M^T -(z - z0)
    normal = MtMinv.matmul(Mt.matmul(-nbhr_diff[..., 2].transpose(1, 0)[:, :, np.newaxis])).squeeze()
    normal = torch.cat([normal, tch_var_f(np.ones((normal.shape[0], 1)))], dim=1).view(pos.shape)

    return normalize(normal)


NORMAL_EST_FN_MAP = {'plane': estimate_surface_normals_plane_fit,
                     'quadric': None,
                     'avg_normal': find_average_normal}


def estimate_surface_normals(pos, kernel_size, method):
    return NORMAL_EST_FN_MAP[method](pos, kernel_size)


def contrast_stretch(im, low=0.01, high=0.099):
    return torch.clamp((im - low) / (high - low), 0.0, 1.0)


def get_normalmap_image(normals, b_normalize=False):
    if b_normalize:
        normals = normalize(normals)
    return np.uint8(normals * 127 + 127)


### Tests


def test_no_cost():
    pos = tch_var_f(np.zeros((5, 5, 3)))
    cost = get_data(spatial_3x3(pos))
    np.testing.assert_equal(cost, 0.0)


def test_plane_estimation_xy_plane():
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(1, -1, 5))
    z = np.ones_like(x) * -1

    pos = tch_var_f(np.stack((x, y, z), axis=2))
    normals = estimate_surface_normals_plane_fit(pos, None)
    normals_ = get_data(normals)
    tmp = normals_.reshape(-1, 3)
    np.testing.assert_equal(tmp, np.array([[0, 0, 1]] * tmp.shape[0]))
    pos_grad = get_data(grad_spatial2d(pos))
    dot_prod = np.sum(pos_grad * normals_[np.newaxis, ...], axis=-1)
    np.testing.assert_array_almost_equal(dot_prod, np.zeros_like(dot_prod))
    nc_cost = get_data(normal_consistency_cost(pos, normals, 1))
    print('nc_cost', nc_cost)
    np.testing.assert_equal(nc_cost, 0.0)


def test_plane_estimation_roty_plane(angle):
    from diffrend.numpy.ops import axis_angle_matrix
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(1, -1, 5))
    z = np.zeros_like(x)

    pos = tch_var_f(np.stack((x, y, z), axis=2))

    M = tch_var_f(axis_angle_matrix(axis=[0, 1, 0], angle=angle))
    pos = pos.view(-1, 3).matmul(M[:, :3].transpose(1, 0))[:, :3].contiguous().view(pos.shape)
    normals = estimate_surface_normals_plane_fit(pos, None)
    normals_ = get_data(normals)
    np.testing.assert_array_almost_equal(np.sum(get_data(pos) * normals_, axis=-1), np.zeros(normals.shape[:2]))
    nc_cost = get_data(normal_consistency_cost(pos, normals, 1))
    print('nc_cost', nc_cost)
    np.testing.assert_almost_equal(nc_cost, 0.0)

    M = tch_var_f(axis_angle_matrix(axis=[0, 1, 0], angle=-angle))
    pos = pos.view(-1, 3).matmul(M[:, :3].transpose(1, 0))[:, :3].contiguous().view(pos.shape)
    normals = estimate_surface_normals_plane_fit(pos, None)
    normals_ = get_data(normals)
    pos_grad = get_data(grad_spatial2d(pos))
    dot_prod = np.sum(pos_grad * normals_[np.newaxis, ...], axis=-1)
    np.testing.assert_array_almost_equal(dot_prod, np.zeros_like(dot_prod))
    nc_cost = get_data(normal_consistency_cost(pos, normals, 1))
    print('nc_cost', nc_cost)
    np.testing.assert_almost_equal(nc_cost, 0.0)


def test_plane_estimation_rotx_plane(angle):
    from diffrend.numpy.ops import axis_angle_matrix
    # rotation about x
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(1, -1, 5))
    z = np.zeros_like(x)

    pos = tch_var_f(np.stack((x, y, z), axis=2))
    M = tch_var_f(axis_angle_matrix(axis=[1, 0, 0], angle=angle))
    pos = pos.view(-1, 3).matmul(M[:, :3].transpose(1, 0))[:, :3].contiguous().view(pos.shape)
    normals = estimate_surface_normals_plane_fit(pos, None)
    normals_ = get_data(normals)
    pos_grad = get_data(grad_spatial2d(pos))
    dot_prod = np.sum(pos_grad * normals_[np.newaxis, ...], axis=-1)
    np.testing.assert_array_almost_equal(dot_prod, np.zeros_like(dot_prod))
    nc_cost = get_data(normal_consistency_cost(pos, normals, 1))
    print('nc_cost', nc_cost)
    np.testing.assert_almost_equal(nc_cost, 0.0)


def test_plane_estimation_rot_plane_range(plane_fn, angle_range, steps):
    for angle in np.linspace(min(angle_range), max(angle_range), steps):
        plane_fn(angle)


def test_plane_estimation_rotx_plane_range(angle_range, steps):
    test_plane_estimation_rot_plane_range(test_plane_estimation_rotx_plane, angle_range, steps)


def test_plane_estimation_roty_plane_range(angle_range, steps):
    test_plane_estimation_rot_plane_range(test_plane_estimation_roty_plane, angle_range, steps)


def test_plane_estimation_on_hemisphere(boundary_eps=0.05):
    """Create a hemisphere in front of a plane and estimate the normals
    on the hemisphere.

    Args:
        boundary_eps: fraction inside from the edge.

    """
    x, y = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(1, -1, 101))
    z = np.sqrt(np.abs(1 - (x ** 2 + y ** 2)))
    out_circle_mask = np.sqrt(x ** 2 + y ** 2) > 1

    z[out_circle_mask] = 0  # make the background a plane

    pos = tch_var_f(np.stack((x, y, z), axis=2))
    normals_gt = get_data(normalize(pos))
    normals = get_data(estimate_surface_normals_plane_fit(pos, None))
    in_circle_mask = np.sqrt(x ** 2 + y ** 2) < (1 - boundary_eps)
    np.testing.assert_array_almost_equal(normals_gt * in_circle_mask[..., np.newaxis],
                                         normals * in_circle_mask[..., np.newaxis])


if __name__ == '__main__':
    test_cam_to_world_identity()
    test_cam_to_world_offset0()
    test_cam_to_world_offset1()
    test_no_cost()
    rand_val = np.random.rand(5, 5, 3)
    cost = spatial_3x3(tch_var_f(rand_val))
    print(rand_val)
    print(cost)
    test_plane_estimation_xy_plane()
    test_plane_estimation_roty_plane(np.pi/4)
    test_plane_estimation_rotx_plane(np.pi/4)
    test_plane_estimation_on_hemisphere(boundary_eps=0.05)
    test_plane_estimation_on_hemisphere(boundary_eps=0.05)
    angle_range_max = np.pi / 2.0 - np.pi / 10.0
    angle_range = [-angle_range_max, angle_range_max]
    test_plane_estimation_rotx_plane_range(angle_range, 100)
    test_plane_estimation_roty_plane_range(angle_range, 100)
