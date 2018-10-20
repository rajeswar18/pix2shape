"""Shapenet dataset."""
import os
import numpy as np
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

    def __getitem__(self, idx):
        """Get item."""
        # Get object path
        obj_path1 = os.path.join(self.opt.root_dir1, 'cube.obj')
        obj_path2 = os.path.join(self.opt.root_dir2, 'sphere_halfbox.obj')
        obj_path3 = os.path.join(self.opt.root_dir3, 'cone.obj')
        # obj_path4 = os.path.join(self.opt.root_dir4, self.samples[idx])

        if not self.loaded:
            self.fg_obj1 = load_model(obj_path1)
            self.fg_obj2 = load_model(obj_path2)
            self.fg_obj3 = load_model(obj_path3)
            # self.fg_obj4 = load_model(obj_path4)
            self.bg_obj = load_model(self.opt.bg_model)
            self.loaded = True
        offset_id=np.random.permutation(4)
        obj_model1 = self.fg_obj1
        obj_model2 = self.fg_obj2
        obj_model3 = self.fg_obj3
        # obj_model4 = self.fg_obj4
        obj2 = self.bg_obj
        v11 = (obj_model1['v'] - obj_model1['v'].mean()) / (obj_model1['v'].max() - obj_model1['v'].min())
        v12 = (obj_model2['v'] - obj_model2['v'].mean()) / (obj_model2['v'].max() - obj_model2['v'].min())
        v13 = (obj_model3['v'] - obj_model3['v'].mean()) / (obj_model3['v'].max() - obj_model3['v'].min())
        # v14 = (obj_model4['v'] - obj_model4['v'].mean()) / (obj_model4['v'].max() - obj_model4['v'].min())
        v2 = obj2['v']  # / (obj2['v'].max() - obj2['v'].min())
        scale = (obj2['v'].max() - obj2['v'].min()) * 0.22
        offset = np.array([[6.9, 6.9, 7.0],[20.4, 6.7, 6.7],[20.4, 6.7, 20.2],[7.0, 6.7, 20.4]]) #+ 2 * np.random.rand(3) #[8.0, 5.0, 18.0],
        if self.opt.only_background:
            v=v2
            f=obj2['f']
        elif self.opt.only_foreground:
            v=v1
            f=obj_model['f']
        else:
            if self.opt.random_rotation:
                random_axis = np_normalize(np.random.rand(3))
                random_angle = np.random.rand(1) * np.pi * 2
                M = axis_angle_matrix(axis=random_axis, angle=random_angle)
                M[:3, 3] = offset[offset_id[0]]+1.5*np.random.randn(3)
                v11 = np.matmul(scale * v11, M.transpose(1, 0)[:3, :3]) + M[:3, 3]
            else:
                # random_axis = np_normalize(np.random.rand(3))
                # random_angle = np.random.rand(1) * np.pi * 2
                # M = axis_angle_matrix(axis=random_axis, angle=random_angle)
                # M[:3, 3] = offset[offset_id[0]]#+1.5*np.random.randn(3)
                # v11 = np.matmul(scale * v11, M.transpose(1, 0)[:3, :3]) + M[:3, 3]
                #
                # random_axis2 = np_normalize(np.random.rand(3))
                # random_angle2 = np.random.rand(1) * np.pi * 2
                # M2 = axis_angle_matrix(axis=random_axis2, angle=random_angle2)
                # M2[:3, 3] = offset[offset_id[2]]#+1.5*np.random.randn(3)
                # v13 = np.matmul(scale * v13, M2.transpose(1, 0)[:3, :3]) + M2[:3, 3]
                v11 = scale * v11 + offset[offset_id[0]]+1.5*np.random.randn(3)
                v12 = scale * v12 + offset[offset_id[1]]+1.5*np.random.randn(3)
                v13 = scale * v13 + offset[offset_id[2]]+1.5*np.random.randn(3)
                # v14 = scale * v14 + offset[offset_id[3]]
            # v = np.concatenate((v11,v12,v13,v14, v2))
            # f = np.concatenate((obj_model1['f'],obj_model2['f']+ v11.shape[0],obj_model3['f']+ v12.shape[0],obj_model4['f']+ v13.shape[0],obj2['f'] + v14.shape[0]))
            v = np.concatenate((v11,v12,v13, v2))
            #import ipdb; ipdb.set_trace()
            f = np.concatenate((obj_model1['f'],obj_model2['f']+ v11.shape[0], obj_model3['f']+ v12.shape[0]+v11.shape[0], obj2['f'] + v13.shape[0]+ v12.shape[0]+v11.shape[0]))


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
