import diffrend.numpy.ops as ops
import numpy as np


def point_along_ray(eye, ray_dir, ray_dist):
    return eye[np.newaxis, np.newaxis, :] + ray_dist[..., np.newaxis] * ray_dir.T[np.newaxis, ...]


def ray_sphere_intersection(eye, ray_dir, sphere):
    """Bundle of rays intersected with a batch of spheres
    :param eye:
    :param ray_dir:
    :param sphere:
    :return:
    """
    pos = sphere['pos']
    pos_tilde = eye - pos
    radius = sphere['radius']

    a = np.sum(ray_dir ** 2, axis=0)
    b = 2 * np.dot(pos_tilde, ray_dir)
    c = (np.sum(pos_tilde ** 2, axis=1) - radius ** 2)[:, np.newaxis]

    d_sqr = b ** 2 - 4 * a * c
    intersect_mask = d_sqr >= 0

    d_sqr = np.where(intersect_mask, d_sqr, np.zeros_like(d_sqr))
    d = np.sqrt(d_sqr)
    inv_denom = 1. / (2 * a)

    t1 = (-b - d) * inv_denom
    t2 = (-b + d) * inv_denom

    # get the nearest positive depth
    t1 = np.where(intersect_mask & (t1 >= 0), t1, np.ones_like(np.max(t1) + 1))
    t2 = np.where(intersect_mask & (t2 >= 0), t2, np.ones_like(np.max(t2) + 1))
    #left_intersect[~intersect_mask] = np.inf
    #right_intersect[~intersect_mask] = np.inf

    ray_dist = np.min(np.stack((t1, t2), axis=2), axis=2)
    ray_dist = np.where(intersect_mask, ray_dist, np.zeros_like(ray_dist))
    #ray_dist[~intersect_mask] = 0  # set this to zero here so that the following line doesn't throw an error
    intersection_pts = point_along_ray(eye, ray_dir, ray_dist)
    #ray_dist[~intersect_mask] = np.inf
    normals = intersection_pts - pos[:, np.newaxis, :]
    normals /= np.sqrt(np.sum(normals ** 2, axis=-1))[..., np.newaxis]
    normals[~intersect_mask] = 0

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersect_mask}


def ray_plane_intersection(eye, ray_dir, plane):
    """Intersection a bundle of rays with a batch of planes
    :param eye: Camera's center of projection
    :param ray_dir: Ray direction
    :param plane: Plane specification
    :return:
    """
    pos = plane['pos']
    normal = ops.normalize(plane['normal'])
    dist = np.sum(pos * normal, axis=1)

    denom = np.dot(normal, ray_dir)

    # check for denom = 0
    intersection_mask = np.abs(denom) > 0

    ray_dist = (dist[:, np.newaxis] - np.dot(normal, eye)[:, np.newaxis]) / denom
    intersection_pts = point_along_ray(eye, ray_dir, ray_dist)
    normals = np.ones_like(intersection_pts) * normal[:, np.newaxis, :]

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


def ray_disk_intersection(eye, ray_dir, disks):
    result = ray_plane_intersection(eye, ray_dir, disks)
    intersection_pts = result['intersect']
    normals = result['normal']
    ray_dist = result['ray_distance']

    centers = disks['pos']

    dist_sqr = np.sum((intersection_pts - centers[:, np.newaxis, :]) ** 2, axis=-1)

    # Intersection mask
    mask_intersect = (dist_sqr <= disks['radius'][:, np.newaxis] ** 2)
    intersection_pts[~mask_intersect] = np.inf
    ray_dist[~mask_intersect] = np.inf

    return {'intersect': intersection_pts, 'normal': normals, 'ray_distance': ray_dist,
            'intersection_mask': mask_intersect}


def ray_triangle_intersection(eye, ray_dir, triangles):
    """Intersection of a bundle of rays with a batch of triangles.
    Assumes that the triangles vertices are specified as F x 3 x 4 matrix where F is the number of faces and
    the normals for all faces are precomputed and in a matrix of size F x 4 (i.e., similar to the normals for other
    geometric primitives). Note that here the number of faces F is the same as number of primitives M.
    :param eye:
    :param ray_dir:
    :param triangles:
    :return:
    """

    planes = {'pos': triangles['face'][:, 0, :], 'normal': triangles['normal']}
    result = ray_plane_intersection(eye, ray_dir, planes)
    intersection_pts = result['intersect']  # M x N x 4 matrix where M is the number of objects and N pixels.
    normals = result['normal'][..., :3]  # M x N x 4
    ray_dist = result['ray_distance']

    # check if intersection point is inside or outside the triangle
    v_p0 = (intersection_pts - triangles['face'][:, 0, :][:, np.newaxis, :])[..., :3]  # M x N x 3
    v_p1 = (intersection_pts - triangles['face'][:, 1, :][:, np.newaxis, :])[..., :3]  # M x N x 3
    v_p2 = (intersection_pts - triangles['face'][:, 2, :][:, np.newaxis, :])[..., :3]  # M x N x 3

    v01 = (triangles['face'][:, 1, :3] - triangles['face'][:, 0, :3])[:, np.newaxis, :]  # M x 1 x 3
    v12 = (triangles['face'][:, 2, :3] - triangles['face'][:, 1, :3])[:, np.newaxis, :]  # M x 1 x 3
    v20 = (triangles['face'][:, 0, :3] - triangles['face'][:, 2, :3])[:, np.newaxis, :]  # M x 1 x 3

    cond_v01 = np.sum(np.cross(v01, v_p0) * normals, axis=-1) >= 0
    cond_v12 = np.sum(np.cross(v12, v_p1) * normals, axis=-1) >= 0
    cond_v20 = np.sum(np.cross(v20, v_p2) * normals, axis=-1) >= 0

    intersection_mask = cond_v01 & cond_v12 & cond_v20
    ray_dist[~intersection_mask] = np.inf

    return {'intersect': intersection_pts, 'normal': result['normal'], 'ray_distance': ray_dist,
            'intersection_mask': intersection_mask}


intersection_fn = {'disk': ray_disk_intersection,
                   'plane': ray_plane_intersection,
                   'sphere': ray_sphere_intersection,
                   'triangle': ray_triangle_intersection,
                   }


def tonemap(im, **kwargs):
    if kwargs['type'] == 'gamma':
        return im ** kwargs['gamma']


def generate_rays(camera):
    viewport = camera['viewport']
    W, H = viewport[2] - viewport[0], viewport[3] - viewport[1]
    aspect_ratio = W / float(H)

    fovy = camera['fovy']
    focal_length = camera['focal_length']
    h = np.tan(fovy / 2) * 2 * focal_length
    w = h * aspect_ratio
    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))

    x *= w / 2
    y *= h / 2

    eye = np.array(camera['eye'])
    ray_dir = np.stack((x.ravel(), y.ravel(), -np.ones(x.size) * focal_length, np.zeros(x.size)), axis=0)
    # view_matrix = lookat(eye=eye, at=camera['at'], up=camera['up'])
    inv_view_matrix = ops.lookat_inv(eye=eye, at=camera['at'], up=camera['up'])
    print(inv_view_matrix, np.linalg.inv(inv_view_matrix))
    ray_dir = np.dot(inv_view_matrix, ray_dir)

    # normalize ray direction
    ray_dir /= np.sqrt(np.sum(ray_dir ** 2, axis=0))

    return eye, ray_dir, H, W


def ray_object_intersections(eye, ray_dir, scene_objects):
    obj_intersections = None
    ray_dist = None
    normals = None
    material_idx = None
    for obj_type in scene_objects:
        result = intersection_fn[obj_type](eye, ray_dir, scene_objects[obj_type])
        curr_intersects = result['intersect']
        curr_ray_dist = result['ray_distance']
        curr_normals = result['normal']
        if curr_ray_dist.ndim == 1:
            curr_ray_dist = curr_ray_dist[np.newaxis, :]
        if curr_intersects.ndim == 1:
            curr_intersects = curr_intersects[np.newaxis, :]
        if curr_normals.ndim == 1:
            curr_normals = curr_normals[np.newaxis, :]

        if obj_intersections is None:
            assert ray_dist is None
            obj_intersections = curr_intersects
            ray_dist = curr_ray_dist
            normals = curr_normals
            material_idx = scene_objects[obj_type]['material_idx']
        else:
            obj_intersections = np.concatenate((obj_intersections, curr_intersects), axis=0)
            ray_dist = np.concatenate((ray_dist, curr_ray_dist), axis=0)
            normals = np.concatenate((normals, curr_normals), axis=0)
            material_idx = np.concatenate((material_idx, scene_objects[obj_type]['material_idx']), axis=0)

    return obj_intersections, ray_dist, normals, material_idx


def render(scene):
    """
    :param scene: Scene description
    :return: [H, W, 3] image
    """
    # Construct rays from the camera's eye position through the screen coordinates
    camera = scene['camera']
    eye, ray_dir, H, W = generate_rays(camera)

    # Ray-object intersections
    scene_objects = scene['objects']
    obj_intersections, ray_dist, normals, material_idx = ray_object_intersections(eye, ray_dir, scene_objects)

    # Valid distances
    pixel_dist = ray_dist
    valid_pixels = (camera['near'] <= ray_dist) & (ray_dist <= camera['far'])
    pixel_dist[~valid_pixels] = np.inf  # Will have to use gather operation for TF and pytorch

    # Nearest object needs to be compared for valid regions only
    nearest_obj = np.argmin(pixel_dist, axis=0)
    C = np.arange(0, nearest_obj.size)  # pixel idx

    # Create depth image for visualization
    # use nearest_obj for gather/select the pixel color
    im_depth = pixel_dist[nearest_obj, C].reshape(H, W)

    ##############################
    # Fragment processing
    ##############################
    # Lighting
    color_table = scene['colors']
    light_pos = scene['lights']['pos']
    light_clr_idx = scene['lights']['color_idx']
    light_colors = color_table[light_clr_idx]

    # Generate the fragments
    """
    Get the normal and material for the visible objects.
    """
    frag_normals = normals[nearest_obj, C]
    frag_pos = obj_intersections[nearest_obj, C]
    frag_albedo = scene['materials']['albedo'][material_idx[nearest_obj]]

    # Fragment shading
    light_dir = light_pos[np.newaxis, :] - frag_pos[:, np.newaxis, :]
    light_dir_norm = np.sqrt(np.sum(light_dir ** 2, axis=-1))[..., np.newaxis]
    light_dir_norm[light_dir_norm <= 0 | np.isinf(light_dir_norm)] = 1
    light_dir = ops.nonzero_divide(light_dir, light_dir_norm)
    im_color = np.sum(frag_normals[:, np.newaxis, :] * light_dir, axis=-1)[..., np.newaxis] * \
               light_colors[np.newaxis, ...] * frag_albedo[:, np.newaxis, :]

    im = np.sum(im_color, axis=1).reshape(H, W, 3)
    im[(im_depth < camera['near']) | (im_depth > camera['far'])] = 0

    # clip negative values
    im[im < 0] = 0

    # Tonemapping
    if 'tonemap' in scene:
        im = tonemap(im, **scene['tonemap'])

    return {'image': im,
            'depth': im_depth,
            'ray_dist': ray_dist,
            'obj_dist': pixel_dist,
            'nearest': nearest_obj.reshape(H, W),
            'ray_dir': ray_dir,
            'valid_pixels': valid_pixels
            }


################
def render_scene(scene):
    res = render(scene)
    im = res['image']

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    plt.imshow(im)
    plt.title('Final Rendered Image')
    plt.savefig('img_np.png')

    depth = res['depth']
    plt.figure()
    plt.imshow(depth)
    plt.title('Depth Image')
    plt.savefig('img_depth_np.png')

    plt.ioff()
    plt.show()
    return res


if __name__ == '__main__':
    scene_basic = {'camera': {'viewport': [0, 0, 320, 240],
                              'fovy': np.deg2rad(90.),
                              'focal_length': 1.,
                              'eye': [0.0, 1.0, 10.0, 1.0],
                              'up': [0.0, 1.0, 0.0, 0.0],
                              'at': [0.0, 0.0, 0.0, 1.0],
                              'near': 1.0,
                              'far': 1000.0,
                              },
                   'lights': {
                       'pos': np.array([[20., 20., 20., 1.0],
                                        [-15, 3., 15., 1.0],
                                        ]),
                       'color_idx': np.array([2, 1]),
                       # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
                       'attenuation': np.array([[0., 1., 0.],
                                                [0., 0., 1.]])
                   },
                   'colors': np.array([[0.0, 0.0, 0.0],
                                       [0.8, 0.1, 0.1],
                                       [0.2, 0.2, 0.2]
                                       ]),
                   'materials': {'albedo': np.array([[0.0, 0.0, 0.0],
                                                     [0.1, 0.1, 0.1],
                                                     [0.2, 0.2, 0.2],
                                                     [0.5, 0.5, 0.5],
                                                     [0.9, 0.1, 0.1],
                                                     [0.1, 0.1, 0.8],
                                                     ])
                                 },
                   'objects': {
                       'disk': {
                           'normal': np.array([[0., 0., 1., 0.0],
                                               [0., 1.0, 0.0, 0.0],
                                               [-1., -1.0, 1., 0.0]]),
                           'pos': np.array([[0., -1., 3., 1.0],
                                            [0., -1., 0, 1.0],
                                            [10., 5., -5, 1.0]]),
                           'radius': np.array([4, 7, 4]),
                           'material_idx': np.array([4, 3, 5])
                       },
                       'sphere': {'pos': np.array([[-8.0, 4.0, -8.0, 1.0],
                                                   [10.0, 0.0, -4.0, 1.0]]),
                                  'radius': np.array([3.0, 2.0]),
                                  'material_idx': np.array([3, 3])
                                  },
                       'triangle': {'face': np.array([[[-20.0, -18.0, -10.0, 1.0],
                                                        [10.0, -18.0, -10.0, 1.],
                                                        [-2.5, 18.0, -10.0, 1.]],
                                                       [[15.0, -18.0, -10.0, 1.0],
                                                        [25, -18.0, -10.0, 1.],
                                                        [20, 18.0, -10.0, 1.]]
                                                       ]),
                                    'normal': np.array([[0., 0., 1., 0.],
                                                        [0., 0., 1., 0.]]),
                                    'material_idx': np.array([5, 4])
                                    }
                   },
                   'tonemap': {'type': 'gamma', 'gamma': 0.8},
                   }
    scene = scene_basic
    # from diffrend.numpy.scene import load_scene, load_mesh_from_file, obj_to_triangle_spec, mesh_to_scene
    #
    # mesh = load_mesh_from_file('../../data/bunny.obj')
    # tri_mesh = obj_to_triangle_spec(mesh)
    #
    # scene = mesh_to_scene(mesh)

    res = render_scene(scene)
