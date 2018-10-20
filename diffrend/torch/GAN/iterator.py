import os
import numpy as np
import PIL.Image as Image
import glob
import pickle as pkl

class Iterator(object):


    def __init__(self, root_path="/home/sai", img_path = 'bamboohouse_final2',
                 cam_path='cam',light_path='light',
                 batch_size=6, nb_sub=None):


        self.root_path = root_path
        self.img_path = os.path.join(root_path, img_path)
        self.campos_path = os.path.join(root_path, cam_path)
        self.lightpos_path = os.path.join(root_path, light_path)
        self.batch_size = batch_size
        self.batch_idx = 0
        self.imgs = glob.glob(self.img_path + "/*light*")


        if nb_sub is not None:
            self.imgs = self.imgs[:nb_sub]




    def _get_img(self, i):

        light_path = self.imgs[i]
        # img = Image.open(img_path)
        # img_array = np.array(img)

        cam_path = light_path[:-10]+"_cam.npy"
        im_path = light_path[:-10]+".npy"

        cam= np.load(cam_path)
        light=np.load(light_path)
        img_array=np.load(im_path)

        return img_array, cam, light

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
