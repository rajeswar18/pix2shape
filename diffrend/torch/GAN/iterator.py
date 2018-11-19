import os
import numpy as np
import glob
import pickle as pkl

from scipy.misc import imread


class Iterator(object):

    # def __init__(self, root_dir="/home/sai", img_path = 'bamboohouse_final2',
    #              cam_path='cam',light_path='light',
    #              batch_size=6, nb_sub=None):

        # self.root_dir = root_dir
        # self.img_path = os.path.join(root_dir, img_path)
        # self.campos_path = os.path.join(root_dir, cam_path)
        # self.lightpos_path = os.path.join(root_dir, light_path)
        # self.batch_size = batch_size
        # self.batch_idx = 0
        # self.imgs = glob.glob(self.img_path + "/*light*")

        # if nb_sub is not None:
        #     self.imgs = self.imgs[:nb_sub]

    def __init__(self, root_dir, batch_size=6, nb_sub=None):

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.batch_idx = 0
        self.image_paths = glob.glob(os.path.join(self.root_dir, '*.png'))
        self.cam_pos = np.loadtxt(os.path.join(self.root_dir, 'cam_pos.csv'), delimiter=',')
        self.light_pos = np.loadtxt(os.path.join(self.root_dir, 'light_00_pos.csv'), delimiter=',')

        if nb_sub is not None:
            self.image_paths = self.image_paths[:nb_sub]

    # def _get_img(self, i):

    #     light_path = self.imgs[i]
    #     # img = Image.open(img_path)
    #     # img_array = np.array(img)

    #     cam_path = light_path[:-10]+"_cam.npy"
    #     im_path = light_path[:-10]+".npy"

    #     cam= np.load(cam_path)
    #     light=np.load(light_path)
    #     img_array=np.load(im_path)

    #     return img_array, cam, light

    def _get_img(self, i):
        return imread(self.image_paths[i % len(self.image_paths)]), self.cam_pos[i % len(self.cam_pos)], self.light_pos[i % len(self.light_pos)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            res = [self[ii] for ii in range(*key.indices(len(self)))]
            xs, ys, caps = zip(*[x for x in res if x is not None])
            return np.array(xs), np.array(ys), np.array(caps)
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                print("The index (%d) is out of range.")
            return self._get_img(key)  # Get the data from elsewhere
        else:
            print("Invalid argument type.")

    def __iter__(self):
        for batch_idx in range(int(len(self)/self.batch_size)):
            if (batch_idx+1)*self.batch_size < len(self):
                yield self[batch_idx*self.batch_size: (batch_idx+1)*self.batch_size]
            else:
                yield self[batch_idx * self.batch_size:]
