import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from diffrend.torch.utils import tensor_dot, estimate_surface_normals_plane_fit, generate_rays

# Imports for test2():
import os
import datetime
from diffrend.torch.GAN.datasets import Dataset_load
from diffrend.torch.GAN.parameters_halfbox_shapenet import Parameters
from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.torch.renderer import render
from diffrend.torch.GAN.main import mkdirs, copy_scripts_to_folder, GAN, create_scene
from diffrend.torch.utils import tch_var_f, tch_var_l, Variable, generate_rays

# NOTE: QUESTIONS:
# - Occlusion in the other architectures? If we don't care, should I then implement this
#   in the same way, where irradiance becomes indirect lighting only?
#   This goes along the point-light discussion. See https://hackmd.io/REemcFBWTZingJYlRPhDMw
# - Are we still training this using through a GAN? We have supervised information now...
# - Is this way of discretizing the hemisphere integral valid? See https://learnopengl.com/PBR/IBL/Diffuse-irradiance

def depth_to_world_coord(depth, view, ray_angles):
    # The camera positions need to be copied for each pixel in the images (add 2 dimensions)
    camera_pos = view[..., None, None, :3]
    camera_dir = view[..., None, None, 3:]

    phi = ray_angles[...,0].unsqueeze(-1)
    theta = ray_angles[...,1].unsqueeze(-1)
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.zeros(*x.size())

    direction = camera_dir + torch.cat((x, y, z), -1)
    direction_normalized = direction / torch.norm(direction, p=2, dim=-1).unsqueeze(-1)
    return camera_pos + depth * direction_normalized


def estimate_normals(world_coords):
    input_wc = world_coords.view(-1, *world_coords.size()[-3:])
    
    # TODO Not batched :(
    normals = []
    for wc in input_wc:
        # It expects a size (width, height, 3) and outputs the same
        normals.append(estimate_surface_normals_plane_fit(wc, None))
    
    outputs = torch.stack(normals)
    return outputs.view(*world_coords.size()[:-3], *outputs.size()[-3:])

    # return world_coords / torch.norm(world_coords, p=2, dim=-1).unsqueeze(-1)

def build_ray_angles(width, height, projection='fisheye'):
    if projection == 'perspective':
        # TODO clean up: use params
        fovy = np.deg2rad(30)
        focal_length = 0.1

        aspect_ratio = width / height
        h = np.tan(fovy / 2) * 2 * focal_length
        w = h * aspect_ratio

        x, y = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(1, -1, height))

        x *= w / 2
        y *= h / 2

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        z = tch_var_f([-focal_length]).repeat(*x.size()[:-1], 1)

        xyz = torch.cat((x, y, z), -1)
        theta = torch.acos(tensor_dot(xyz, tch_var_f([0, 0, -1]).repeat(*xyz.size()[:-1], 1), axis=-1).unsqueeze(-1))
        phi = torch.atan2(x, y)
        return torch.cat((phi, theta), -1)
    elif projection == 'fisheye':
        phi, theta = torch.meshgrid(torch.linspace(0, 2*np.pi, width), torch.linspace(0, np.pi, height))
        phi = phi.unsqueeze(-1)
        theta = theta.unsqueeze(-1)
        return torch.cat((phi, theta), -1)
    else:
        raise ValueError(f'Unsupported projection {projection}')

def build_rays(view, width, height, projection='fisheye'):
    # If the tensor has more than 2 dimensions, "fold" everything in the 'batch' dimension
    input_view = view.view(-1, view.size(-1))
    print("******* BUILD RAYS *******")
    # NOTE: This method seems to be the biggest bottleneck so far. It NEEDS to be batched
    print(view.size())

    # TODO build x,y,z here and pass as argument to generate_rays()

    # TODO Not batched :(
    ray_dirs = []
    for i, v in enumerate(input_view):
        if i % 100 == 0:
            print(f"Building rays {i} / {input_view.size(0)}")

        # TODO clean up (use params)
        camera = {
            'viewport': [0, 0, width, height],
            'fovy': np.deg2rad(30),
            'focal_length': 0.1,
            'eye': v[:3],
            'at': v[:3] + v[3:],
            'up': tch_var_f([0, 1, 0]),
            'proj_type': projection
        }
        _, ray_dir, _, _ = generate_rays(camera)
        ray_dirs.append(ray_dir)

    outputs = torch.stack(ray_dirs)
    print(outputs.view(*view.size()[:-1], width, height, -1).size())
    return outputs.view(*view.size()[:-1], width, height, -1)


class SurfelsModel(nn.Module):
    """
    Combination of depth and local properties prediction
    """
    def __init__(self, render): # TODO remove
        super(SurfelsModel, self).__init__()
        self.render = render

    def forward(self, z, view, ray_angles, projection='perspective'):
        # TODO: Use theta, phi pairs instead of 3D rays. I believe this would help

        # If the input tensors have more than 2 dimensions, "fold" everything in the 'batch' dimension
        input_view = view.view(-1, view.size(-1))
        # Notice that the rays are also folded in the same way, since each ray can be computed independently
        input_ray_angles = ray_angles.view(-1, ray_angles.size(-1))

        print("SurfelsModel")
        # print(input.size())

        # TODO NN

        # output = torch.zeros(*rays.size()[:-1], 7)
        # depth = output[..., :1]
        # albedo = output[..., 1:4]
        # emittance = output[..., 4:]

        # TODO remove
        depth = self.render(view, ray_angles.size(-3), ray_angles.size(-2), projection=projection)
        albedo = tch_var_f([0.5, 0.5, 0.5]).repeat(*ray_angles.size()[:-1], 1)
        emittance = tch_var_f([0.1, 0.1, 0.1]).repeat(*ray_angles.size()[:-1], 1)

        # unfold the result
        return depth, albedo, emittance
        # return output.view(*view.size()[:-1], *output.size()[-3:])


class RootIrradiance(nn.Module):
    def __init__(self, width, height):
        super(RootIrradiance, self).__init__()
        self.width = width
        self.height = height

    def forward(self, z, view, ray_angles=None):
        # print("RootIrradiance")
        # print(view.size())
        # TODO is rays necessary here? If so, remove '=None' in params
        return torch.rand(*view.size()[:-1], self.width, self.height, 3), []
        # return torch.zeros(*view.size()[:-1], 1, 1, 3)


class Renderer(nn.Module):
    def __init__(self, render=None, width=128, height=128, level=0, max_level=1, surfels_model=None):
        super(Renderer, self).__init__()

        self.width = width
        self.height = height
        self.level = level  # Recursion level
        self.surfels_model = surfels_model if surfels_model is not None else SurfelsModel(render)

        # TODO pixel-wise russian roulette?
        next_width = max(1, width // 8)
        next_height = max(1, height // 8)
        if level < max_level:
            self.next_renderer = Renderer(
                width=next_width, height=next_height, level=level+1, max_level=max_level,
                surfels_model=self.surfels_model)
        else:
            self.next_renderer = RootIrradiance(next_width, next_height)

    def forward(self, z, view, ray_angles=None):
        """
        z: latent code. Shape: (batch, z_size)
        view: camera views. Shape: (batch, n, n, n//8, n//8, n//16, ..., 6) depending on which self.level we are at
        view[...,:3] is expected to be the camera position and view[...,3:] to be the direction in which the camera is looking
        """
        projection = 'fisheye'
        if ray_angles is None:
            # For the first recursion level, we want to output a planar image, not a fisheye representation
            # in order to be able to directly compare it to the input image
            ray_angles = build_ray_angles(self.width, self.height,
                            projection='perspective')
            projection = 'perspective'

        # Local properties needs to change based on whether we are rendering a fisheye view or a planar view
        # For this reason, it also needs to be passed the rays
        depth, albedo, emittance = self.surfels_model(z, view,
                                    ray_angles, projection=projection)

        world_coords = depth_to_world_coord(depth, view, ray_angles)
        normals = estimate_normals(world_coords)

        # The new camera views. Place a camera at each surfel position, looking in the direction of the normal
        next_views = torch.cat([world_coords, normals], -1)
        next_ray_angles = build_ray_angles(self.next_renderer.width,
                        self.next_renderer.height, projection='fisheye')
        irradiance, saved_outputs = self.next_renderer(
            z, next_views, ray_angles=next_ray_angles)  # Pass incident_directions to avoid a recomputation

        incident_cosines = torch.cos(next_ray_angles[...,1].unsqueeze(-1)) # cos(theta_i)
        incident_sines = torch.sin(next_ray_angles[...,1].unsqueeze(-1)) # sin(theta_i)
        n = self.next_renderer.width * self.next_renderer.height

        # Rendering equation for a discretized irradiance with lambertian diffuse surfaces:
        # See https://learnopengl.com/PBR/IBL/Diffuse-irradiance
        print(f"Min in incident_cosines: {torch.min(incident_cosines)}, Max: {torch.max(incident_cosines)}, Avg: {torch.mean(incident_cosines)}")
        print(f"Min in incident_sines: {torch.min(incident_sines)}, Max: {torch.max(incident_sines)}, Avg: {torch.mean(incident_sines)}")
        print(f"Min in irradiance: {torch.min(irradiance)}, Max: {torch.max(irradiance)}, Avg: {torch.mean(irradiance)}")
        whatever = irradiance * incident_cosines * incident_sines
        print(whatever.size())
        print(f"Min in whatever: {torch.min(whatever)}, Max: {torch.max(whatever)}, Avg: {torch.mean(whatever)}")

        output_image = emittance + albedo / n * \
            torch.sum(whatever, (-2, -3))

        # Concatenate the results from the next recursion level with these
        # These are the outputs that we can run a loss through to train the networks
        saved_outputs.append({
            'view': view,
            'output_image': output_image,
            'depth': depth,
            'albedo': albedo,
            'emittance': emittance
        })
        return output_image, saved_outputs


def test():
    r = Renderer()
    batch_size = 4
    z = torch.zeros(batch_size, 256)
    view = torch.zeros(batch_size, 6)
    output_image, saved_outputs = r(z, view)
    print(f"Output image size: {output_image.size()}")
    print(f"Saved output dimensionality: {len(saved_outputs)}")
    for i, d in enumerate(reversed(saved_outputs)):
        print(f"View size at level {i}: {d['view'].size()}")

def test2():
    """Start training."""
    # Parse args
    opt = Parameters().parse()

    scene = create_scene(opt.width, opt.height,
                    opt.fovy, opt.focal_length,
                    opt.n_splats)

    if 'sphere' in scene['objects']:
        del scene['objects']['sphere']
    if 'disk' in scene['objects']:
        del scene['objects']['disk']
    if 'triangle' not in scene['objects']:
        scene['objects'] = {
            'triangle': {'face': None, 'normal': None,
                            'material_idx': None}}

    dataset_loader = Dataset_load(opt)
    dataset_loader.initialize_dataset()
    dataset_loader.initialize_dataset_loader(1)
    samples = iter(dataset_loader.get_dataset_loader()).next()

    scene['objects']['triangle']['material_idx'] = tch_var_l(
        np.zeros(samples['mesh']['face'][0].shape[0],
                    dtype=int).tolist())
    scene['objects']['triangle']['face'] = Variable(
        samples['mesh']['face'][0], requires_grad=False)
    scene['objects']['triangle']['normal'] = Variable(
        samples['mesh']['normal'][0],
        requires_grad=False)
    
    cam_pos = uniform_sample_sphere(
                    radius=opt.cam_dist, num_samples=opt.batchSize,
                    axis=None, angle=None,
                    theta_range=np.deg2rad(opt.theta),
                    phi_range=np.deg2rad(opt.phi))
    
    scene['camera']['at'] = tch_var_f([0.05, 0.0, 0.0])
    scene['camera']['eye'] = tch_var_f(cam_pos[0])

    res = render(scene,
                norm_depth_image_only=opt.norm_depth_image_only,
                double_sided=True, use_quartic=opt.use_quartic)

    # Also TODO: It seems like the returned depth might be scaled or something?
    # The world_coords computed seem to be all over the place -> TEST

    plt.imshow(res['image'] / 255)
    plt.show()

    def render_new_view(view, width, height, projection='perspective'):
        input_views = view.view(-1, view.size(-1))
        outputs = []

        scene['camera']['viewport'] = [0, 0, width, height]
        scene['camera']['proj_type'] = projection

        for i, v in enumerate(input_views):
            if i % 100 == 0:
                print(f"Rendering {i} / {input_views.size(0)}")

            scene['camera']['eye'] = v[:3]
            scene['camera']['at'] = v[:3] + v[3:]
            # TODO change render so that it also returns albedo and emittance
            outputs.append(render(scene,
                    norm_depth_image_only=opt.norm_depth_image_only,
                    double_sided=True, use_quartic=opt.use_quartic)['depth'].unsqueeze(-1))

        outputs = torch.stack(outputs)
        return outputs.view(*view.size()[:-1], *outputs.size()[-3:])

    r = Renderer(render_new_view, max_level=1)
    z = torch.zeros(256)
    view_dir = (scene['camera']['at'] - scene['camera']['eye']) / \
        torch.norm(scene['camera']['at'] - scene['camera']['eye'], p=2)
    view = torch.cat([scene['camera']['eye'], view_dir])
    output_image, saved_outputs = r(z, view)
    print(f"Output image size: {output_image.size()}")
    print(f"Saved output dimensionality: {len(saved_outputs)}")
    for i, d in enumerate(reversed(saved_outputs)):
        print(f"View size at level {i}: {d['view'].size()}")

    print(f"Min in image: {torch.min(output_image)}, Max in image: {torch.max(output_image)}")
    plt.imshow(output_image)
    plt.show()


if __name__ == '__main__':
    test2()
