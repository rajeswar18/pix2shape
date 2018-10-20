from diffrend.model import load_obj, load_off, load_splat, obj_to_triangle_spec
import numpy as np
import json


def load_scene(filename):
    with open(filename, 'r') as fp:
        scene = json.load(fp)
    return scene


def write_scene(scene, filename):
    with open(filename, 'w') as fp:
        json.dump(scene, fp)

load_fn = {'obj': load_obj,
           'off': load_off,
           'splat': load_splat,
           }


def load_mesh_from_file(filename):
    import os

    prefix, ext = os.path.splitext(filename)
    mesh = load_fn[ext[1:]](filename)
    return mesh


def setup_mesh(mesh, material, path_prefix=None):
    tri_mesh = obj_to_triangle_spec(mesh, material=None)
    return tri_mesh


def mesh_to_scene(mesh, **params):
    tri_mesh = obj_to_triangle_spec(mesh)
    if 'camera' in params:
        camera = params['camera']
    else:
        camera = {"eye": [0.0, 0.0, 10.0, 1.0],
                   "near": 1.0,
                   "focal_length": 1.0,
                   "far": 1000.0,
                   "fovy": 1.57079,
                   "viewport": [0, 0, 320, 240],
                   "up": [0.0, 1.0, 0.0, 0.0],
                   "at": [0.0, 0.0, 0.0, 1.0]
                  }
    if 'materials' in params:
        materials = params['materials']
    else:
        materials = {"albedo": [[0.0, 0.0, 0.0],
                                 [0.1, 0.1, 0.1],
                                 [0.2, 0.2, 0.2],
                                 [0.5, 0.5, 0.5],
                                 [0.9, 0.1, 0.1],
                                 [0.1, 0.1, 0.8]]
                      }
    if 'colors' in params:
        colors = params['colors']
    else:
        colors = [[0.0, 0.0, 0.0],
                   [0.8, 0.1, 0.1],
                   [0.2, 0.2, 0.2]],

    if 'lights' in params:
        lights = params['lights']
    else:
        lights = {
                       'pos': np.array([[20., 20., 20., 1.0],
                                        [-15, 3., 15., 1.0],
                                        ]),
                       'color_idx': np.array([2, 1]),
                       # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
                       'attenuation': np.array([[0., 1., 0.],
                                                [0., 0., 1.]])
                   },

    mesh_scene = {'camera': camera,
                   'lights': lights,
                   'colors': colors,
                   'materials': materials,
                   'objects': {
                       'triangle': tri_mesh
                   },
                   'tonemap': {'type': 'gamma', 'gamma': 0.8},
                 }
    return mesh_scene

class Scene(object):
    def __init__(self, scene=None):
        import os
        if '.json' in scene:
            path_prefix, filename = os.path.split(scene)
            self.path_prefix = path_prefix
            scene = load_scene(scene)

        self.scene = scene
        self.camera = scene['camera']
        self.objects = scene['objects']
        self.lights = scene['lights']
        self.materials = scene['materials']
        self.colors = np.array(scene['colors'])
        self.tonemap = scene['tonemap'] if 'tonemap' in self.scene else False
        self.__setup()

    def __getattr__(self, item):
        return self.scene[item]

    def __setup(self):
        self.camera['eye'] = np.array(self.camera['eye'])
        self.camera['up'] = np.array(self.camera['up'])
        self.camera['at'] = np.array(self.camera['at'])
        self.materials['albedo'] = np.array(self.materials['albedo'])
        self.lights['pos'] = np.array(self.lights['pos'])
        self.lights['attenuation'] = np.array(self.lights['attenuation'])

        # Make object pos and normals numpy arrays
        # ...

        # Load geometry from file as needed
        if 'mesh' in self.scene:
            pass



def test_scene_write_and_load():
    scene_v0 = {'camera': {'viewport': [0, 0, 320, 240],
                              'fovy': 1.57079,
                              'focal_length': 1.,
                              'eye': [0.0, 0.0, 10.0, 1.0],
                              'up': [0.0, 1.0, 0.0, 0.0],
                              'at': [0.0, 0.0, 0.0, 1.0],
                              'near': 1.0,
                              'far': 1000.0,
                              },
             'lights': {
                 'pos': [[20., 20., 20., 1.0],
                         [-15, 3., 15., 1.0]],
                 'color_idx': [2, 1],
                 # Light attenuation factors have the form (kc, kl, kq) and eq: 1/(kc + kl * d + kq * d^2)
                 'attenuation': [[0., 1., 0.],
                                 [0., 0., 1.]]
             },
             'colors': [[0.0, 0.0, 0.0],
                        [0.8, 0.1, 0.1],
                        [0.2, 0.2, 0.2]
                        ],
             'materials': {'albedo': [[0.0, 0.0, 0.0],
                                      [0.1, 0.1, 0.1],
                                      [0.2, 0.2, 0.2],
                                      [0.5, 0.5, 0.5],
                                      [0.9, 0.1, 0.1],
                                      [0.1, 0.1, 0.8],
                                      ]
                           },
             'objects': {
                 'disk': {
                     'normal': [[0., 0., 1., 0.0],
                                [0., 1.0, 0.0, 0.0],
                                [-1., -1.0, 1., 0.0]],
                     'pos': [[0., -1., 3., 1.0],
                             [0., -1., 0, 1.0],
                             [10., 5., -5, 1.0]],
                     'radius': [4, 7, 4],
                     'material_idx': [4, 3, 5]
                 },
                 'sphere': {'pos': [[-8.0, 4.0, -8.0, 1.0],
                                    [10.0, 0.0, -4.0, 1.0]],
                            'radius': [3.0, 2.0],
                            'material_idx': [3, 3]
                 },
             },
             'tonemap': {'type': 'gamma', 'gamma': 0.8},
            }
    write_scene(scene_v0, 'test_scene.json')
    test_scene = load_scene('test_scene.json')
    match = test_scene == scene_v0
    if not match:
        print('Test failed')

    return match

def test_scene_loading(filename):
    pass

if __name__ == '__main__':
    scene = load_scene('../../data/basic_scene.json')
    bunny_scene = load_scene('../../data/bunny_scene.json')

    mesh = load_mesh_from_file('../../data/bunny.obj')
    tri_mesh = obj_to_triangle_spec(mesh)

    mesh_scene = mesh_to_scene(mesh)
    write_scene(mesh_scene, 'bunny_scene.json')




