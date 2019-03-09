import torch
from torch import nn
import torch.optim as optim
import numpy as np

from diffrend.torch.utils import tensor_f, tensor_l, device, tensor_dot, estimate_surface_normals_plane_fit

# Imports for test2():
import itertools
from imageio import imwrite
from diffrend.torch.GAN.datasets import Dataset_load
from diffrend.torch.GAN.parameters_halfbox_shapenet import Parameters
from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.torch.renderer import render
from diffrend.torch.GAN.main import create_scene, gauss_reparametrize
from diffrend.torch.GAN.twin_networks import LatentEncoder, weights_init

# NOTE: QUESTIONS:
# - Occlusion in the other architectures? If we don't care, should I then implement this
#   in the same way, where irradiance becomes indirect lighting only?
#   This goes along the point-light discussion. See https://hackmd.io/REemcFBWTZingJYlRPhDMw
# - Are we still training this using a GAN? We have supervised information now... A: Start with L2 loss. Then try GAN for regularization properties
# - Would estimate_surface_normals_plane_fit() work with a fisheye projection??


def depth_to_world_coord(depth, view, ray_angles):
    # print("Depth to world")
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
    # print("Estimate Normals")
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
    # print("Build ray angles")
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

    def __init__(self, latent_size):
        super(SurfelsModel, self).__init__()
        self.latent_size = latent_size

        self.network = nn.Sequential(
            nn.Linear(latent_size + 5, 30),
            nn.BatchNorm1d(30),
            nn.LeakyReLU(0.01, True),
            nn.Linear(30, 15),
            nn.BatchNorm1d(15),
            nn.LeakyReLU(0.01, True),
            nn.Linear(15, 1),
            nn.ReLU(True) # Make sure the depth output is positive
        )

    def forward(self, z, view, ray_angles, projection='perspective'):
        # print(f"SurfelsModel {ray_angles.size(-3)}x{ray_angles.size(-2)}")

        # If the input tensors have more than 2 dimensions, "fold" everything in the 'batch' dimension
        input_view = view.view(-1, view.size(-1))
        input_z = z.view(-1, z.size(-1))
        # Notice that the rays are also folded in the same way, since each ray can be computed independently
        input_ray_angles = ray_angles.view(-1, ray_angles.size(-1))

        input_camera_pos = input_view[...,:3].repeat(input_ray_angles.size(0), 1)
        input_ray_angles = input_ray_angles.repeat(input_view.size(0), 1)
        input_z = input_z.repeat(int(input_camera_pos.size(0) / input_z.size(0)), 1)

        input = torch.cat((input_z, input_camera_pos, input_ray_angles), -1)

        output = self.network(input)

        # unfold the result
        output = output.view(*view.size()[:-1], *ray_angles.size()[-3:-1], 1)

        depth = output[..., :1]
        # albedo = output[..., 1:4]
        # emittance = output[..., 1:]

        # TODO remove
        albedo = tensor_f([0.5, 0.5, 0.5]).repeat(*view.size()[:-1], *ray_angles.size()[:-1], 1)
        emittance = tensor_f([0., 0., 0.]).repeat(*view.size()[:-1], *ray_angles.size()[:-1], 1)

        return depth, albedo, emittance


class RootIrradiance(nn.Module):
    def __init__(self, width, height, latent_size):
        super(RootIrradiance, self).__init__()
        self.width = width
        self.height = height

        # self.network2 = nn.Sequential(
        #     nn.Upsample(size=(self.width, self.height), mode='bilinear'),
        #     nn.Conv2d(100 + 6, 32, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.01, True),
        #     nn.Conv2d(32, 3, 3, padding=1, bias=False)
        # )

        self.network = nn.Sequential(
            nn.Linear(latent_size + 5, 30),
            nn.BatchNorm1d(30),
            nn.LeakyReLU(0.01, True),
            nn.Linear(30, 15),
            nn.BatchNorm1d(15),
            nn.LeakyReLU(0.01, True),
            nn.Linear(15, 3),
            nn.ReLU(True) # Make sure the depth output is positive
        )

    def forward(self, z, view, ray_angles=None):
        # print(f"RootIrradiance {self.width}x{self.height}")
        # print(view.size())
        # TODO is rays necessary here? If so, remove '=None' in params
        # input_view = view.view(-1, view.size(-1), 1, 1)
        # input_z = z.view(-1, z.size(-1), 1, 1)
        # input_z = input_z.repeat(int(input_view.size(0) / input_z.size(0)), 1, 1, 1)
        # input = torch.cat((input_z, input_view), 1)

        # output = self.network2(input)

        # # Unfold the output and move the channel dimension back to the end
        # output = output.view(*view.size()[:-1], *output.size()[-2:], output.size(-3))


        # TODO: This is almost the exact same as in SurfelsModel. Merge the two
        input_view = view.view(-1, view.size(-1))
        input_z = z.view(-1, z.size(-1))
        # Notice that the rays are also folded in the same way, since each ray can be computed independently
        input_ray_angles = ray_angles.view(-1, ray_angles.size(-1))

        input_camera_pos = input_view[...,:3].repeat(input_ray_angles.size(0), 1)
        input_ray_angles = input_ray_angles.repeat(input_view.size(0), 1)
        input_z = input_z.repeat(int(input_camera_pos.size(0) / input_z.size(0)), 1)

        input = torch.cat((input_z, input_camera_pos, input_ray_angles), -1)

        output = self.network(input)

        # Unfold the output and move the channel dimension back to the end
        output = output.view(*view.size()[:-1], *ray_angles.size()[-3:-1], 3)

        return output, []


class Renderer(nn.Module):
    def __init__(self, width=128, height=128, decay=8, min_size=[4, 4],
                 level=0, max_level=1, surfels_model=None,
                 latent_size=100):
        super(Renderer, self).__init__()

        self.width = width
        self.height = height
        self.level = level  # Recursion level
        self.surfels_model = surfels_model if surfels_model is not None else SurfelsModel(latent_size)

        # TODO pixel-wise russian roulette?
        next_width = max(min_size[0], width // decay)
        next_height = max(min_size[1], height // decay)
        if level < max_level:
            self.next_renderer = Renderer(width=next_width, height=next_height,
                                          decay=decay, min_size=min_size,
                                          level=level+1, max_level=max_level,
                                          surfels_model=self.surfels_model,
                                          latent_size=latent_size)
        else:
            self.next_renderer = RootIrradiance(next_width, next_height, latent_size)

    def forward(self, z, view, ray_angles=None):
        """
        z: latent code. Shape: (batch, z_size)
        view: camera views. Shape: (batch, n, n, n//8, n//8, n//16, ..., 6) depending on which self.level we are at
        view[...,:3] is expected to be the camera position and view[...,3:] to be the (normalized) direction in which the camera is looking
        """
        # print(f"Renderer {self.width}x{self.height}")
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
        # print(
        #     f"Min in incident_cosines: {torch.min(incident_cosines)}, Max: {torch.max(incident_cosines)}, Avg: {torch.mean(incident_cosines)}")
        # print(
        #     f"Min in incident_sines: {torch.min(incident_sines)}, Max: {torch.max(incident_sines)}, Avg: {torch.mean(incident_sines)}")
        print(f"Min in irradiance: {torch.min(irradiance)}, Max: {torch.max(irradiance)}, Avg: {torch.mean(irradiance)}")
        # whatever = irradiance * incident_cosines * incident_sines
        # print(f"Size of whatever: {whatever.size()}")
        # print(
        #     f"Min in whatever: {torch.min(whatever)}, Max: {torch.max(whatever)}, Avg: {torch.mean(whatever)}")

        output_image = emittance + albedo * np.pi / n * \
            torch.sum(irradiance * incident_cosines * incident_sines, (-2, -3))

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


def test1():
    batch_size = 2
    latent_size = 100
    width = height = 64
    r = Renderer(latent_size=latent_size, width=width, height=height)
    z = torch.zeros(batch_size, latent_size, device=device)
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

    opt.width = opt.height = 64  # TODO remove
    opt.cam_dist = 0.8
    opt.lr = 0.001
    decay = 8
    latent_size = 100
    max_level = 0
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
    dataset_loader = dataset_loader.get_dataset_loader()
    data_iter = iter(dataset_loader)

    # Define the NN stuff
    encoder = LatentEncoder(latent_size, 3, opt.nef)
    renderer = Renderer(max_level=max_level, decay=decay, latent_size=latent_size,
                 width=opt.width, height=opt.height)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(itertools.chain(encoder.parameters(), renderer.parameters()),
                           lr=opt.lr,
                           betas=(opt.beta1, 0.999))

    # Init the weights:
    encoder.apply(weights_init)
    renderer.apply(weights_init)

    # Set the networks to training mode and put them on the right device
    encoder.train()
    renderer.train()
    encoder.to(device)
    renderer.to(device)

    num_samples = 100
    for i in range(num_samples):
        print(f"Example: {i} / {num_samples-1}")
        data_iter = iter(dataset_loader) # Why do we need to do this every time?
        samples = data_iter.next()

        scene['objects']['triangle']['material_idx'] = tensor_l(
            np.zeros(samples['mesh']['face'][0].shape[0], dtype=int).tolist())
        scene['objects']['triangle']['face'] = samples['mesh']['face'][0].to(device)
        scene['objects']['triangle']['normal'] = samples['mesh']['normal'][0].to(device)

        cam_pos = uniform_sample_sphere(
            radius=opt.cam_dist, num_samples=1,
            axis=None, angle=None,
            theta_range=np.deg2rad(opt.theta),
            phi_range=np.deg2rad(opt.phi))
        
        # TODO randomize light

        scene['camera']['at'] = tensor_f([0.05, 0.0, 0.0])
        scene['camera']['eye'] = tensor_f(cam_pos[0])
        cam_dir = scene['camera']['at'] - scene['camera']['eye']
        cam_dir = cam_dir / torch.norm(cam_dir, p=2, dim=-1).unsqueeze(-1)

        res = render(scene,
                    norm_depth_image_only=opt.norm_depth_image_only,
                    double_sided=True, use_quartic=opt.use_quartic)

        # put the channel dimension before the width and height
        input_image = res['image'].view(1, res['image'].size(-1), *res['image'].size()[:-1])

        mu_z, logvar_z = encoder(input_image)
        z = gauss_reparametrize(mu_z, logvar_z)

        view = torch.cat((scene['camera']['eye'], cam_dir), -1).unsqueeze(0)

        # Run and train
        optimizer.zero_grad()
        output_image, saved_outputs = renderer(z.view(-1, latent_size), view)
        output_image = output_image.squeeze(0)

        print(f"Min in image: {torch.min(output_image)}, Max: {torch.max(output_image)}, Avg: {torch.mean(output_image)}")

        output_image_normalized = (output_image - torch.min(output_image)) /\
                (torch.max(output_image) - torch.min(output_image) + 1e-10)
                
        loss = criterion(output_image_normalized, res['image'])
        loss.backward()
        optimizer.step()
        # TODO add other losses on the saved_outputs

        print(f"Loss at step {i}: {loss}")

        # Log outputs
        folder = 'tmp_outputs'
        imwrite(f'{folder}/input_image_{i}.png', res['image'].cpu().numpy())
        imwrite(f'{folder}/input_image_normalized_{i}.png', ((res['image'] - torch.min(res['image'])) /
                (torch.max(res['image']) - torch.min(res['image']) + 1e-10)).cpu().numpy())
        imwrite(f'{folder}/output_image_{i}.png', output_image.detach().cpu().numpy())
        imwrite(f'{folder}/output_image_normalized_{i}.png', output_image_normalized.detach().cpu().numpy())


    # def render_new_view(view, width, height, projection='perspective'):
    #     input_views = view.view(-1, view.size(-1))
    #     depths = []
    #     images = []

    #     scene['camera']['viewport'] = [0, 0, width, height]
    #     scene['camera']['proj_type'] = projection

    #     print(f"Rendering with sizes {width}, {height}")

    #     for i, v in enumerate(input_views):
    #         if i % 100 == 0:
    #             print(f"Rendering {i} / {input_views.size(0)}")

    #         scene['camera']['eye'] = v[:3]
    #         scene['camera']['at'] = v[:3] + v[3:]
    #         # TODO change render so that it also returns albedo and emittance

    #         render_result = render(scene,
    #                                norm_depth_image_only=opt.norm_depth_image_only,
    #                                double_sided=True, use_quartic=opt.use_quartic)
    #         depths.append(render_result['depth'].unsqueeze(-1))
    #         images.append(render_result['image'])

    #     depths = torch.stack(depths)
    #     images = torch.stack(images)
    #     return {
    #         'depth': depths.view(*view.size()[:-1], *depths.size()[-3:]),
    #         'image': images.view(*view.size()[:-1], *images.size()[-3:])
    #     }


if __name__ == '__main__':
    test2()
