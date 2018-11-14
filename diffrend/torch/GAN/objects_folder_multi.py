"""Shapenet dataset."""
import os
import numpy as np
import random
from torch.utils.data import Dataset
from diffrend.model import load_model, obj_to_triangle_spec
from diffrend.utils.sample_generator import uniform_sample_mesh
from diffrend.numpy.ops import axis_angle_matrix
from diffrend.numpy.ops import normalize as np_normalize


class ObjectsFolderMultiObjectDataset(Dataset):
    """Objects folder dataset."""

    def __init__(self, opt, transform=None):
        """Constructor.

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.opt = opt
        self.transform = transform
        self.n_samples = 0
        self.samples = []
        self.loaded = False
        self.bg_obj = None
        self.fg_obj = None
        # Get object paths

        # Get object paths
        self._get_objects_paths()
        print ("Total samples: {}".format(len(self.samples)))

        # self.scene = self._create_scene()

    def __len__(self):
        """Get dataset length."""
        return len(self.samples)

    def genAttrSentences(self, object_type, obj_size, location, possible_sentences):
        s = "There is a "+obj_size+" "+object_type+" located at the "+location
        possible_sentences.append(s)
        return possible_sentences

    def genCamSentences(self, object_type, cam_dist, fovy, possible_sentences):
        location = "too close to the camera"       
        if cam_dist == 0.8:
            location = "too close to the camera"
        elif cam_dist == 1.2:
            location = "neither too close nor far away from the camera"
        elif cam_dist == 1.6:
            location = "far away from the camera"

        s = "There is a "+object_type+" located "+location
        possible_sentences.append(s)

        focus = "focussed too highly"
        if fovy == 26:
            focus = "focussed too highly"
        elif fovy == 30:
            focus = "focussed correctly"
        elif fovy == 34:
            focus = "focussed too less"

        s1 = "There is a "+object_type+" that is "+focus
        possible_sentences.append(s1)

        return possible_sentences


    def genPossibleSentences(self, object_type, scale_coef, scale_factors, offset, cam_dist, fovy):
        possible_sentences = []
        obj_size="small"


        if scale_factors.index(scale_coef) == 0:
            obj_size = "small"
        elif scale_factors.index(scale_coef) == 1:
            obj_size = "medium"
        elif scale_factors.index(scale_coef) == 2:
            obj_size = "large"

        location = "middle of the room"
        if offset == 0:
            location = "right corner of the room"
        elif offset == 1:
            location = "top corner of the room"
        elif offset == 2:
            location = "left corner of the room"
        elif offset == 3:
            location = "middle of the right part of the room"
        elif offset == 4:
            location = "middle of the bottom part of the room"
        elif offset == 5:
            location = "middle of the left part of the room"
        elif offset == 6:
            location = "middle of the room"

        ## Generate sentences based on scale and offset parameters
        possible_sentences = self.genAttrSentences(object_type, obj_size, location, possible_sentences)

        ## Generate sentences based on camera position and fovy
        possible_sentences = self.genCamSentences(object_type, cam_dist, fovy, possible_sentences)

        return possible_sentences


    def __getitem__(self, idx):
        """Get item."""
        # Get object path
        obj_path = os.path.join(self.opt.root_dir, self.samples[idx])

        if not self.loaded:
            self.fg_obj = load_model(obj_path)
            self.bg_obj = load_model(self.opt.bg_model)
            self.loaded = True
        obj_model = self.fg_obj
        obj2 = self.bg_obj
        v1 = (obj_model['v'] - obj_model['v'].mean()) / (obj_model['v'].max() - obj_model['v'].min())
        v2 = obj2['v']  # / (obj2['v'].max() - obj2['v'].min())
        
        #print('scale coeff is ', scale_coef)
        #scale = (obj2['v'].max() - obj2['v'].min()) * scale_coef[0][1]
        #scale = (obj2['v'].max() - obj2['v'].min()) * 0.3
        #print('obj2 v max is ', obj2['v'].max())
        #print('obj2 v min is ', obj2['v'].min())

        if self.samples[idx].split(".")[0] == "cube":
            scale_factors = [0.1, 0.2, 0.3]
            scale_coef = random.sample(list(enumerate(scale_factors)), 1)
            if scale_coef[0][1] == 0.1:
                possible_offsets = [np.array([30.0, 8.0, 12.0]), np.array([14.0, 30.0, 12.0]), np.array([14.0, 8.0, 30.0]), np.array([25.0, 25.0, 12.0]), np.array([25.0, 8.0, 25.0]), np.array([14.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)
            elif scale_coef[0][1] == 0.2:
                possible_offsets = [np.array([28.0, 10.0, 12.0]), np.array([14.0, 28.0, 12.0]), np.array([14.0, 10.0, 28.0]), np.array([25.0, 25.0, 12.0]), np.array([25.0, 10.0, 25.0]), np.array([14.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)
            elif scale_coef[0][1] == 0.3:
                possible_offsets = [np.array([28.0, 10.0, 12.0]), np.array([14.0, 28.0, 12.0]), np.array([14.0, 10.0, 28.0]), np.array([25.0, 25.0, 12.0]), np.array([25.0, 10.0, 25.0]), np.array([14.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)

        elif self.samples[idx].split(".")[0] == "cone":
            #print("inside cone ....")
            scale_factors = [0.2, 0.3, 0.4]
            scale_coef = random.sample(list(enumerate(scale_factors)), 1)
            if scale_coef[0][1] == 0.2:
                possible_offsets = [np.array([35.0, 8.0, 8.0]), np.array([8.0, 35.0, 8.0]), np.array([8.0, 8.0, 35.0]), np.array([25.0, 25.0, 8.0]), np.array([25.0, 8.0, 25.0]), np.array([8.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)
            elif scale_coef[0][1] == 0.3:
                possible_offsets = [np.array([35.0, 8.0, 8.0]), np.array([8.0, 35.0, 8.0]), np.array([8.0, 8.0, 35.0]), np.array([25.0, 25.0, 8.0]), np.array([25.0, 8.0, 25.0]), np.array([8.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)
            elif scale_coef[0][1] == 0.4:
                possible_offsets = [np.array([35.0, 8.0, 8.0]), np.array([8.0, 35.0, 8.0]), np.array([8.0, 8.0, 35.0]), np.array([25.0, 25.0, 8.0]), np.array([25.0, 8.0, 25.0]), np.array([8.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)

        elif self.samples[idx].split(".")[0] == "sphere":
            #print("inside sphere ...... ")
            scale_factors = [0.1, 0.2, 0.3]
            scale_coef = random.sample(list(enumerate(scale_factors)), 1)
            if scale_coef[0][1] == 0.1:
                possible_offsets = [np.array([30.0, 8.0, 12.0]), np.array([12.0, 30.0, 12.0]), np.array([12.0, 8.0, 30.0]), np.array([25.0, 25.0, 12.0]), np.array([25.0, 8.0, 25.0]), np.array([12.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)
            elif scale_coef[0][1] == 0.2:
                possible_offsets = [np.array([30.0, 8.0, 12.0]), np.array([12.0, 30.0, 12.0]), np.array([12.0, 8.0, 30.0]), np.array([25.0, 25.0, 12.0]), np.array([25.0, 8.0, 25.0]), np.array([12.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)
            elif scale_coef[0][1] == 0.3:
                possible_offsets = [np.array([30.0, 8.0, 14.0]), np.array([14.0, 30.0, 14.0]), np.array([14.0, 8.0, 30.0]), np.array([25.0, 25.0, 14.0]), np.array([25.0, 8.0, 25.0]), np.array([14.0, 25.0, 25.0]), np.array([25.0, 25.0, 25.0])]
                chosen_offset = random.sample(list(enumerate(possible_offsets)), 1)


        scale = (obj2['v'].max() - obj2['v'].min()) * scale_coef[0][1]
        #print("scale coef is ", scale_coef[0][1])
        offset = chosen_offset[0][1]
        #print("chosen offset is ", offset)

        # scale = (obj2['v'].max() - obj2['v'].min()) * 0.3
        # print("scale coef is ", scale_coef[0][1])
        # offset = np.array([25.0, 8.0, 25.0])
        # print("chosen offset is ", offset)

        # generates possible sentences
        arr = self.genPossibleSentences(self.samples[idx].split(".")[0], scale_coef[0][1], scale_factors, chosen_offset[0][0], self.opt.cam_dist, self.opt.fovy)
        print('possible sentences ', arr)

        if self.opt.only_background:
            v=v2
            f=obj2['f']
        elif self.opt.only_foreground:
            v=v1
            f=obj_model['f']
        else:
            if self.opt.random_rotation:
                random_axis = np_normalize(self.opt.axis)
                random_angle = np.random.rand(1) * np.pi * 2
                M = axis_angle_matrix(axis=random_axis, angle=random_angle)
                M[:3, 3] = offset
                v1 = np.matmul(scale * v1, M.transpose(1, 0)[:3, :3]) + M[:3, 3]
            else:
                v1 = scale * v1 + offset
            v = np.concatenate((v1, v2))
            f = np.concatenate((obj_model['f'], obj2['f'] + v1.shape[0]))

        obj_model = {'v': v, 'f': f}

        if self.opt.use_mesh:
            # normalize the vertices
            v = obj_model['v']
            axis_range = np.max(v, axis=0) - np.min(v, axis=0)
            v = (v - np.mean(v, axis=0)) / max(axis_range)  # Normalize to make the largest spread 1
            obj_model['v'] = v
            mesh = obj_to_triangle_spec(obj_model)
            meshes = {'face': mesh['face'].astype(np.float32),
                      'normal': mesh['normal'].astype(np.float32)}
            sample = {'synset': 0, 'mesh': meshes}
        else:
            # Sample points from the 3D mesh
            v, vn = uniform_sample_mesh(obj_model,
                                        num_samples=self.opt.n_splats)
            # Normalize the vertices
            v = (v - np.mean(v, axis=0)) / (v.max() - v.min())

            # Save the splats
            splats = {'pos': v.astype(np.float32),
                      'normal': vn.astype(np.float32)}

            # Add model and synset to the output dictionary
            sample = {'synset': 0, 'splats': splats}

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_objects_paths(self,):
        print (self.opt.root_dir)
        for o in os.listdir(self.opt.root_dir):
            self.samples.append(o)


def main():
    """Test function."""
    dataset = ObjectsFolderDataset(
        root_dir='/home/dvazquez/datasets/shapenet/ShapeNetCore.v2',
        transform=None)
    print (len(dataset))

    for f in dataset:
        print (f)


if __name__ == "__main__":
    main()
