import glob
import numpy as np
import os

from scipy.misc import imread


class Iterator(object):

    def __init__(self, root_dir, batch_size=6, shuffle=False, nb_sub=None):

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_idx = 0

        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'img',  '*.png')))
        self.depth_image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'depth',  '*.png')))
        self.cam_pos = np.loadtxt(os.path.join(self.root_dir, 'cam_pos.csv'), delimiter=',')
        self.light_pos = np.loadtxt(os.path.join(self.root_dir, 'light_00_pos.csv'), delimiter=',')

        if nb_sub is not None:
            self.image_paths = self.image_paths[:nb_sub]
            self.depth_image_paths = self.depth_image_paths[:nb_sub]
            self.cam_pos = self.cam_pos[:nb_sub]
            self.light_pos = self.light_pos[:nb_sub]

        # Indices for returning batches, possibly shuffled order of images
        self.idx = np.arange(len(self))

        if self.shuffle:
            np.random.shuffle(self.idx)

        # Current index for batches
        self.batch_idx = 0

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, image_path):
        return imread(image_path)

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch_idx + self.batch_size - 1 > len(self):
            # Reset
            self.batch_idx = 0
            # Shuffle indices again
            if self.shuffle:
                np.random.shuffle(self.idx)

        # Return batch of images, cam_pos, light_pos
        batch_images = [imread(self.image_paths[idx])[:, :, :3] for idx in self.idx[self.batch_idx:self.batch_idx+self.batch_size]]
        batch_depth_images = [imread(self.depth_image_paths[idx])[:, :, :3] for idx in self.idx[self.batch_idx:self.batch_idx+self.batch_size]]
        batch_cam_pos = [self.cam_pos[idx] for idx in self.idx[self.batch_idx:self.batch_idx+self.batch_size]]
        # batch_light_pos = [self.light_pos[idx] for idx in self.idx[self.batch_idx:self.batch_idx+self.batch_size]]

        self.batch_idx += self.batch_size

        return batch_images, batch_depth_images, batch_cam_pos
