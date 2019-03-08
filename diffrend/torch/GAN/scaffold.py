import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from diffrend.torch.utils import tensor_dot, estimate_surface_normals_plane_fit, generate_rays

# Imports for test2():
import os
import datetime
from imageio import imwrite
from diffrend.torch.GAN.datasets import Dataset_load
from diffrend.torch.GAN.parameters_halfbox_shapenet import Parameters
from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.torch.renderer import render
from diffrend.torch.GAN.main import mkdirs, copy_scripts_to_folder, GAN, create_scene
from diffrend.torch.utils import tensor_f, tensor_l, device, generate_rays, lookat_rot_inv

# NOTE: QUESTIONS:
# - Occlusion in the other architectures? If we don't care, should I then implement this
#   in the same way, where irradiance becomes indirect lighting only?
#   This goes along the point-light discussion. See https://hackmd.io/REemcFBWTZingJYlRPhDMw
# - Are we still training this using a GAN? We have supervised information now... A: Start with L2 loss. Then try GAN for regularization properties
# - Would estimate_surface_normals_plane_fit() work with a fisheye projection??


def depth_to_world_coord(depth, view, ray_angles):
    # The camera positions need to be copied for each pixel in the images (add 2 dimensions)
    camera_pos = view[..., None, None, :3]
    camera_dir = view[..., None, None, 3:]

    camera_up = tensor_f([0, 1, 0]).repeat(
        *camera_dir.size()[:-1], 1)  # TODO use params

    phi = ray_angles[..., 0].unsqueeze(-1)  # + phi_cam_dir.unsqueeze(-1)
    theta = ray_angles[..., 1].unsqueeze(-1)  # + theta_cam_dir.unsqueeze(-1)
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)

    # vector perpendicular to camera_dir and camera_up
    p = torch.cross(camera_dir, camera_up, -1)
    direction = x * p + y * camera_up + z * camera_dir  # This should have norm 1
    return camera_pos + depth * direction  # direction_normalized


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

        x, y = torch.meshgrid(
            torch.linspace(-w / 2, w / 2, width, device=device), torch.linspace(-h / 2, h / 2, height, device=device))

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        z = tensor_f([-focal_length]).repeat(*x.size()[:-1], 1)

        xyz = torch.cat((x, y, z), -1)
        xyz_normalized = xyz / torch.norm(xyz, p=2, dim=-1).unsqueeze(-1)
        # z = cos(theta)
        theta = torch.acos(-xyz_normalized[..., 2].unsqueeze(-1))
        phi = torch.atan2(x, y)
    elif projection == 'fisheye':
        # For some reason torch.linspace does not work with a step count of 1...
        phi = torch.linspace(
            0, 2 * np.pi, width, device=device) if width > 1 else tensor_f([np.pi])
        theta = torch.linspace(
            0, np.pi / 2, height, device=device) if height > 1 else tensor_f([np.pi / 4])
        phi, theta = torch.meshgrid(phi, theta)
        phi = phi.unsqueeze(-1)
        theta = theta.unsqueeze(-1)
    else:
        raise ValueError(f'Unsupported projection {projection}')

    return torch.cat((phi, theta), -1)


class SurfelsModel(nn.Module):
    """
    Combination of depth and local properties prediction
    """

    def __init__(self, render):  # TODO remove
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
        # if len(view.size()) > 2:
        #     view = view[60:70, 60:70].contiguous()
        res = self.render(view, ray_angles.size(-3),
                          ray_angles.size(-2), projection=projection)
        depth = res['depth']

        # images = res['image']
        # images = images.view(-1, *images.size()[-3:])
        # for i, im in enumerate(images):
        #     print(f"{i} / {len(images)}")
        #     print(view.view(-1, view.size(-1))[i])
        #     plt.imshow(im / 255)
        #     plt.show()

        albedo = tensor_f([0.5, 0.5, 0.5]).repeat(*ray_angles.size()[:-1], 1)
        emittance = tensor_f([0., 0., 0.]).repeat(
            *ray_angles.size()[:-1], 1)

        # unfold the result
        return depth, albedo, emittance
        # return output.view(*view.size()[:-1], *output.size()[-3:])


class RootIrradiance(nn.Module):
    def __init__(self, width, height, render):
        super(RootIrradiance, self).__init__()
        self.width = width
        self.height = height
        self.render = render  # TODO remove

    def forward(self, z, view, ray_angles=None):
        # print("RootIrradiance")
        # print(view.size())
        # TODO is rays necessary here? If so, remove '=None' in params
        images = self.render(view, ray_angles.size(-3),
                             ray_angles.size(-2), projection='fisheye')['image']
        return images / 255, []

        # return torch.rand(*view.size()[:-1], self.width, self.height, 3), []
        # return torch.zeros(*view.size()[:-1], 1, 1, 3)


class Renderer(nn.Module):
    def __init__(self, render, width=128, height=128, decay=8, level=0, max_level=1, surfels_model=None):
        super(Renderer, self).__init__()

        self.width = width
        self.height = height
        self.level = level  # Recursion level
        self.surfels_model = surfels_model if surfels_model is not None else SurfelsModel(
            render)

        # TODO pixel-wise russian roulette?
        next_width = max(1, width // decay)
        next_height = max(1, height // decay)
        if level < max_level:
            self.next_renderer = Renderer(render, width=next_width,
                                          height=next_height, decay=decay,
                                          level=level+1, max_level=max_level,
                                          surfels_model=self.surfels_model)
        else:
            self.next_renderer = RootIrradiance(
                next_width, next_height, render)

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

        incident_cosines = torch.cos(
            next_ray_angles[..., 1].unsqueeze(-1))  # cos(theta_i)
        incident_sines = torch.sin(
            next_ray_angles[..., 1].unsqueeze(-1))  # sin(theta_i)
        n = self.next_renderer.width * self.next_renderer.height

        # Rendering equation for a discretized irradiance with lambertian diffuse surfaces:
        # See https://learnopengl.com/PBR/IBL/Diffuse-irradiance
        # and http://www.codinglabs.net/article_physically_based_rendering.aspx
        print(
            f"Min in incident_cosines: {torch.min(incident_cosines)}, Max: {torch.max(incident_cosines)}, Avg: {torch.mean(incident_cosines)}")
        print(
            f"Min in incident_sines: {torch.min(incident_sines)}, Max: {torch.max(incident_sines)}, Avg: {torch.mean(incident_sines)}")
        print(
            f"Min in irradiance: {torch.min(irradiance)}, Max: {torch.max(irradiance)}, Avg: {torch.mean(irradiance)}")
        whatever = irradiance * incident_cosines * incident_sines
        print(f"Size of whatever: {whatever.size()}")
        print(
            f"Min in whatever: {torch.min(whatever)}, Max: {torch.max(whatever)}, Avg: {torch.mean(whatever)}")

        output_image = emittance + albedo * np.pi / n * \
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
    z = torch.zeros(batch_size, 256, device=device)
    view = torch.zeros(batch_size, 6, device=device)
    output_image, saved_outputs = r(z, view)
    print(f"Output image size: {output_image.size()}")
    print(f"Saved output dimensionality: {len(saved_outputs)}")
    for i, d in enumerate(reversed(saved_outputs)):
        print(f"View size at level {i}: {d['view'].size()}")


def test2():
    """Start training."""
    # Parse args
    opt = Parameters().parse()

    opt.width, opt.height = 128, 128  # TODO remove
    opt.cam_dist = 0.8
    decay = 8
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

    scene['objects']['triangle']['material_idx'] = tensor_l(
        np.zeros(samples['mesh']['face'][0].shape[0], dtype=int).tolist())
    scene['objects']['triangle']['face'] = samples['mesh']['face'][0].to(device)
    scene['objects']['triangle']['normal'] = samples['mesh']['normal'][0].to(device)

    cam_pos = uniform_sample_sphere(
        radius=opt.cam_dist, num_samples=1,
        axis=None, angle=None,
        theta_range=np.deg2rad(opt.theta),
        phi_range=np.deg2rad(opt.phi))

    scene['camera']['at'] = tensor_f([0.05, 0.0, 0.0])
    scene['camera']['eye'] = tensor_f(cam_pos[0])
    # scene['camera']['eye'] = tch_var_f([opt.cam_dist, 0, 0])

    res = render(scene,
                 norm_depth_image_only=opt.norm_depth_image_only,
                 double_sided=True, use_quartic=opt.use_quartic)

    imwrite('original_render.png',
            (res['image'] / torch.max(res['image'])).cpu().numpy())
    # plt.imshow(res['image'] / torch.max(res['image']))
    # plt.show()

    def render_new_view(view, width, height, projection='perspective'):
        input_views = view.view(-1, view.size(-1))
        depths = []
        images = []

        scene['camera']['viewport'] = [0, 0, width, height]
        scene['camera']['proj_type'] = projection

        print(f"Rendering with sizes {width}, {height}")

        for i, v in enumerate(input_views):
            if i % 100 == 0:
                print(f"Rendering {i} / {input_views.size(0)}")

            scene['camera']['eye'] = v[:3]
            scene['camera']['at'] = v[:3] + v[3:]
            # TODO change render so that it also returns albedo and emittance

            render_result = render(scene,
                                   norm_depth_image_only=opt.norm_depth_image_only,
                                   double_sided=True, use_quartic=opt.use_quartic)
            depths.append(render_result['depth'].unsqueeze(-1))
            images.append(render_result['image'])

        depths = torch.stack(depths)
        images = torch.stack(images)
        return {
            'depth': depths.view(*view.size()[:-1], *depths.size()[-3:]),
            'image': images.view(*view.size()[:-1], *images.size()[-3:])
        }

    r = Renderer(render_new_view, max_level=1, decay=decay,
                 width=opt.width, height=opt.height)
    z = torch.zeros(256, device=device)
    view_dir = (scene['camera']['at'] - scene['camera']['eye']) / \
        torch.norm(scene['camera']['at'] - scene['camera']['eye'], p=2)
    view = torch.cat([scene['camera']['eye'], view_dir])
    output_image, saved_outputs = r(z, view)
    print(f"Output image size: {output_image.size()}")
    print(f"Saved output dimensionality: {len(saved_outputs)}")
    for i, d in enumerate(reversed(saved_outputs)):
        print(f"View size at level {i}: {d['view'].size()}")

    print(
        f"Min in image: {torch.min(output_image)}, Max in image: {torch.max(output_image)}")

    # TODO this normalization seems to make a HUGE difference since I observed that this max
    # can be as low as 0.02... probably because we do not make use of direct light
    imwrite('output_image.png', (output_image /
                                 torch.max(output_image)).cpu().numpy())
    imwrite('output_image2.png', ((output_image - torch.min(output_image)) /
                                 (torch.max(output_image) - torch.min(output_image))).cpu().numpy())
    imwrite('output_image_unscaled.png', output_image.cpu().numpy())
    # plt.imshow(output_image / torch.max(output_image))
    # plt.show()


if __name__ == '__main__':
    test2()
