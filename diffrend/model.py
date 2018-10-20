"""Load 3D model related functions."""
from __future__ import absolute_import
import numpy as np
import re
import diffrend.numpy.ops as ops


def norm_sqr(v):
    """Compute the sqrtnorm of a vector."""
    return np.sum(v ** 2, axis=-1)


def norm(v):
    """Compute the norm of a vector."""
    return np.sqrt(norm_sqr(v))


def compute_face_normal(obj, unnormalized=False):
    """Compute the normal of a mesh face."""
    v0 = obj['v'][obj['f'][:, 0]]
    v1 = obj['v'][obj['f'][:, 1]]
    v2 = obj['v'][obj['f'][:, 2]]

    v01 = v1 - v0
    v02 = v2 - v0

    n = np.cross(v01, v02)
    if unnormalized:
        return n
    denom = norm(n)[..., np.newaxis]
    denom[denom == 0] = 1
    return n / denom


def compute_circum_circle(obj):
    """Compute Circunference circle.

    https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_from_cross-_and_dot-products
    :param obj:
    :return: {'center': circle_center, 'radius': circle_radius}
    """
    p1 = obj['v'][obj['f'][:, 0]]
    p2 = obj['v'][obj['f'][:, 1]]
    p3 = obj['v'][obj['f'][:, 2]]

    p23 = p2 - p3
    p13 = p1 - p3
    p12 = p1 - p2

    n = np.cross(p12, p23)
    n_norm_sqr = norm_sqr(n)

    radius = norm(p12) * norm(p23) * norm(-p13) / (2 * np.sqrt(n_norm_sqr))

    denom_inv = 1 / (2 * n_norm_sqr)
    alpha = norm_sqr(p23) * np.sum(p12 * p13, axis=-1) * denom_inv
    beta = norm_sqr(p13) * np.sum(-p12 * p23, axis=-1) * denom_inv
    gamma = norm_sqr(p12) * np.sum(-p13 * -p23, axis=-1) * denom_inv

    center = (alpha[:, np.newaxis] * p1 + beta[:, np.newaxis] *
              p2 + gamma[:, np.newaxis] * p3)

    return {'center': center, 'radius': radius}


def obj_to_splat(obj, use_circum_circle=True, camera=None):
    """Convert meshes to splats."""
    if not use_circum_circle:
        raise ValueError('obj_to_splat only supports circumscribed circle.')
    if camera is not None:
        # Back-face culling: remove faces pointing away from the camera
        obj = ops.backface_culling(obj, camera)
    cc = compute_circum_circle(obj)
    normals = compute_face_normal(obj)
    return {'v': cc['center'], 'r': cc['radius'], 'vn': normals}


def write_splat(filename, obj):
    """Write 3D m odel in splat format to a file."""
    with open(filename, 'w') as f:
        for v, vn, r in zip(obj['v'], obj['vn'], obj['r']):
            f.write('v {}\n'.format(' '.join([str(x) for x in v])))
            f.write('vn {}\n'.format(' '.join([str(x) for x in vn])))
            if type(r) is np.ndarray or type(r) is list:
                f.write('r {}\n'.format(' '.join([str(x) for x in r])))
            else:
                f.write('r {}\n'.format(r))


def load_splat(filename, verbose=True):
    """Load 3D model in splats format."""
    with open(filename, 'r') as f:
        obj_file = f.read().splitlines()

    v = []  # list of vertices
    vn = []  # list of vertex normals
    r = []   # list of splat radii single value means disk, pair ellipse, triple ellipsoid/sphere

    for line in obj_file:
        line = re.sub(r'\s+', ' ', line).lstrip().rstrip().split(' ')
        if line[0] == '#':
            continue
        if line[0] == 'v':
            v.append([float(x) for x in line[1:]])
        if line[0] == 'vn':
            vn.append([float(x) for x in line[1:]])
        if line[0] == 'r':
            r.append([float(x) for x in line[1:]])

    if verbose:
        print('Vertex count: {}'.format(len(v)))
        print('Vertex Normal count: {}'.format(len(vn)))
        print('Splat Radius count: {}'.format(len(r)))

    return {'v': np.array(v), 'vn': np.array(vn), 'r': np.array(r),
            'type': 'splat'}


def load_obj(filename, verbose=True):
    """Read .obj file."""
    with open(filename, 'r') as f:
        obj_file = f.read().splitlines()
    # print(obj_file)

    v = []  # list of vertices
    f = []  # list of faces

    for line in obj_file:
        # line = line.split(' ')
        line = re.sub(r'\s+', ' ', line).lstrip().rstrip().split(' ')
        if line[0] == '#':
            continue
        if line[0] == 'v':
            v.append([float(x) for x in line[1:]])
        if line[0] == 'f':
            # Note the conversion from 1-based to 0-based index
            # Note skipping normals and texture index and avoiding extra spaces
            f.append([int(x.split('/')[0]) - 1 for x in line[1:] if x])

    if verbose:
        print('Vertex count: {}'.format(len(v)))
        print('Face count: {}'.format(len(f)))

    return {'v': np.array(v), 'f': np.array(f)}


def load_off(filename, verbose=True):
    """Load 3D model in .off format."""
    with open(filename, 'r') as f:
        obj_file = f.read().splitlines()

    v = []  # list of vertices
    f = []  # list of faces
    e = []  # list of edges

    num_vertices = None
    num_faces = None
    num_edges = None
    check_num_vertex_face = False

    for line in obj_file:
        line = re.sub(r'\s+', ' ', line).lstrip().rstrip().split(' ')
        if line[0] == 'OFF':
            check_num_vertex_face = True
            if len(line) > 1:
                num_vertices, num_faces, num_edges = int(line[1]), int(line[2]), int(line[3])
                check_num_vertex_face = False
            continue
        elif check_num_vertex_face:
            num_vertices, num_faces, num_edges = int(line[0]), int(line[1]), int(line[2])
            check_num_vertex_face = False
            continue

        if len(v) < num_vertices:
            v.append([float(x) for x in line])
        elif len(f) < num_faces:
            f.append([int(x) for x in line[1:]])
        elif len(e) < num_edges:
            e.append([int(x) for x in line])

    if verbose:
        print('#V: {}, #F: {}, #E: {}'.format(num_vertices, num_faces,
                                              num_edges))
        print('Vertex count: {}'.format(len(v)))
        print('Face count: {}'.format(len(f)))
        print('Edge count: {}'.format(len(e)))

    return {'v': np.array(v), 'f': np.array(f), 'e': np.array(e)}


def load_model(filename, verbose=False):
    """Load 3D model. Accepts .off .obj .splat."""
    import os
    prefix, ext = os.path.splitext(filename)
    model_loader_fn = {'off': load_off,
                       'obj': load_obj,
                       'splat': load_splat}

    return model_loader_fn[ext[1:]](filename, verbose)


def obj_to_triangle_spec(obj):
    """Object to triangle specs."""
    faces = obj['v'][obj['f']]
    normals = compute_face_normal(obj)
    faces = np.concatenate((
        faces, np.ones_like(faces[..., 0])[..., np.newaxis]), axis=-1)
    normals = np.concatenate((
        normals, np.zeros_like(normals[..., 0])[..., np.newaxis]), axis=-1)

    return {'face': faces, 'normal': normals}


def test_run():
    """Test."""
    filename = '../data/bunny.obj'
    obj_data = load_obj(filename)
    cc = compute_circum_circle(obj_data)
    filename = '../data/desk_0007.off'
    off_data = load_off(filename)

    return obj_data, off_data


if __name__ == '__main__':
    obj_data, off_data = test_run()
    obj_splat = obj_to_splat(obj_data)
    write_splat('test.splat', obj_splat)
    splat_obj = load_splat('test.splat')
    tri_bunny = obj_to_triangle_spec(obj_data)
    tri_desk = obj_to_triangle_spec(off_data)
