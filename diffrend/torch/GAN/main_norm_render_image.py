"""Generator."""
from __future__ import absolute_import

import sys
sys.path.append('../../../')

import matplotlib
matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import axes3d

import copy
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision

from imageio import imsave
# from scipy.misc import imsave
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from diffrend.torch.GAN.datasets import Dataset_load
from diffrend.torch.GAN.iterator import Iterator
from diffrend.torch.GAN.parameters_halfbox_shapenet import Parameters
from diffrend.torch.GAN.twin_networks import create_networks
from diffrend.torch.params import SCENE_SPHERE_HALFBOX_0
from diffrend.torch.renderer import (render, render_splats_along_ray,
                                     z_to_pcl_CC)
from diffrend.torch.utils import (tch_var_f, tch_var_l, get_data,
                                  get_normalmap_image, cam_to_world,
                                  unit_norm2_L2loss, normal_consistency_cost,
                                  away_from_camera_penalty, spatial_3x3,
                                  depth_rgb_gradient_consistency,
                                  grad_spatial2d)
from diffrend.utils.sample_generator import uniform_sample_sphere
from diffrend.utils.utils import contrast_stretch_percentile, save_xyz


"""
 z_noise -> z_distance_generator -> proj_to_scene_in_CC ->
 normal_estimator_network -> render -> discriminator
 real_data -> discriminator
"""

def mem_check():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("Mem:", process.memory_info().rss/1024/1024/1024, "GB")


def copy_scripts_to_folder(expr_dir):
    """Copy scripts."""
    shutil.copy("iterator.py", expr_dir)
    shutil.copy("datasets.py", expr_dir)
    shutil.copy("objects_folder_multi.py", expr_dir)
    shutil.copy("parameters_halfbox_shapenet.py", expr_dir)
    shutil.copy("twin_networks.py", expr_dir)
    shutil.copy("../params.py", expr_dir)
    shutil.copy("../renderer.py", expr_dir)
    shutil.copy(__file__, expr_dir)


def mkdirs(paths):
    """Create paths."""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """Create a directory."""
    if not os.path.exists(path):
        os.makedirs(path)


def create_scene(width, height, fovy, focal_length, n_samples):
    """Create a semi-empty scene with camera parameters."""
    # Create a splats rendering scene
    scene = copy.deepcopy(SCENE_SPHERE_HALFBOX_0)

    # Define the camera parameters
    scene['camera']['viewport'] = [0, 0, width, height]
    scene['camera']['fovy'] = np.deg2rad(fovy)
    scene['camera']['focal_length'] = focal_length

    return scene

def gauss_reparametrize(mu, logvar, n_sample=1):
    """Gaussian reparametrization"""
    std = logvar.mul(0.5).exp_()
    size = std.size()
    eps = Variable(std.data.new(size[0], n_sample, size[1]).normal_())
    z = eps.mul(std[:, None, :]).add_(mu[:, None, :])
    z = torch.clamp(z, -4., 4.)
    return z.view(z.size(0)*z.size(1), z.size(2), 1, 1)


def calc_gradient_penalty(discriminator, encoder, real_data, fake_data, fake_data_cond, z, z_enc,
                          gp_lambda, no_cuda):
    """Calculate GP."""
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    if not no_cuda:
        alpha = alpha.cuda()

    interpolates = Variable(alpha * real_data + ((1 - alpha) * fake_data),
                            requires_grad=True)
    interpolate_mu_z, interpolate_logvar_z = encoder(interpolates)

    interpolate_z = gauss_reparametrize(interpolate_mu_z, interpolate_logvar_z)

    interpolates_cond = Variable(fake_data_cond, requires_grad=True)
    disc_interpolates = discriminator(interpolates, interpolates_cond, interpolate_z.detach())
    if not no_cuda:
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    else:
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty


class GAN(object):
    """GAN class."""

    def __init__(self, opt, dataset_load=None, exp_dir=None):
        """Constructor."""
        # Save variables
        self.opt = opt
        self.dataset_load = dataset_load
        self.opt.out_dir = exp_dir

        # Define other variables
        self.real_label = 1
        self.fake_label = 0

        # Losses file
        file_name = os.path.join(self.opt.out_dir, 'render_batch_losses.txt')
        self.output_loss_file = open(file_name, "wt")

        # TODO: Add comment
        if self.opt.full_sphere_sampling:
            self.opt.phi = None
            self.opt.theta = None
            self.opt.cam_dist = self.opt.cam_dist + 0.2
        else:
            self.opt.angle = None
            #self.opt.axis = None

        # TensorboardX
        self.writer = SummaryWriter(self.opt.vis_monitoring)
        print(self.opt.vis_monitoring)
        print(self.opt.out_dir)
        # Create dataset loader
        self.create_dataset_loader()

        # Create the networks
        self.create_networks()

        # Create create_tensors
        self.create_tensors()

        # Create criterion
        self.create_criterion()

        # Create create optimizers
        self.create_optimizers()

        # Create splats rendering scene
        self.create_scene()

    def create_dataset_loader(self, ):
        """Create dataset leader."""
        # Define camera positions
        if self.opt.same_view:
            # self.cam_pos = uniform_sample_sphere(radius=self.opt.cam_dist,
            #                                      num_samples=1)
            arrays = [np.asarray([3., 3., 3.]) for _ in
                      range(self.opt.batchSize)]  # TODO: Magic numbers
            self.cam_pos = np.stack(arrays, axis=0)

        # Create dataset loader
        self.dataset_load.initialize_dataset()
        self.dataset = self.dataset_load.get_dataset()
        self.dataset_load.initialize_dataset_loader(1)  # TODO: Hack
        self.dataset_loader = self.dataset_load.get_dataset_loader()

    def create_networks(self, ):
        """Create networks."""
        self.netG, _, self.netD, self.netD2, self.netE = create_networks(
            self.opt, verbose=True, depth_only=True)  # TODO: Remove D2 and G2
        # Create the normal estimation network which takes pointclouds in the
        # camera space and outputs the normals

        self.sph_normals = True

        if not self.opt.no_cuda:
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()
            self.netE = self.netE.cuda()


    def create_scene(self, ):
        """Create a semi-empty scene with camera parameters."""
        self.scene = create_scene(
            self.opt.splats_img_size, self.opt.splats_img_size, self.opt.fovy,
            self.opt.focal_length, self.opt.n_splats)

    def create_tensors(self, ):
        """Create the tensors."""
        # Create tensors
        self.input = torch.FloatTensor(
            self.opt.batchSize, self.opt.render_img_nc,
            self.opt.render_img_size, self.opt.render_img_size)
        self.input_depth = torch.FloatTensor(
            self.opt.batchSize, 1, self.opt.render_img_size,
            self.opt.render_img_size)
        self.input_normal = torch.FloatTensor(
            self.opt.batchSize, 1, self.opt.render_img_size,
            self.opt.render_img_size)
        self.input_cond = torch.FloatTensor(self.opt.batchSize, 3)

        self.noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1)
        self.fixed_noise = torch.FloatTensor(
            self.opt.batchSize, int(self.opt.nz), 1, 1).normal_(0, 1)

        self.label = torch.FloatTensor(2*self.opt.batchSize)
        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1

        # Move them to the GPU
        if not self.opt.no_cuda:
            self.input = self.input.cuda()
            self.input_depth = self.input_depth.cuda()
            self.input_normal = self.input_normal.cuda()
            self.input_cond = self.input_cond.cuda()

            self.label = self.label.cuda()
            self.noise = self.noise.cuda()
            self.fixed_noise = self.fixed_noise.cuda()

            self.one = self.one.cuda()
            self.mone = self.mone.cuda()

        self.fixed_noise = Variable(self.fixed_noise)  # TODO: Why?

    def create_criterion(self, ):
        """Create criterion."""
        self.criterion = nn.BCELoss()
        if not self.opt.no_cuda:
            self.criterion = self.criterion.cuda()

    def create_optimizers(self, ):
        """Create optimizers."""
        if self.opt.optimizer == 'adam':
            self.optimizerD = optim.Adam(self.netD.parameters(),
                                         lr=self.opt.lr,
                                         betas=(self.opt.beta1, 0.999))
            self.optimizerG = optim.Adam(itertools.chain(self.netG.parameters(), self.netE.parameters()),
                                         lr=self.opt.lr,
                                         betas=(self.opt.beta1, 0.999))

        elif self.opt.optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(self.netD.parameters(),
                                            lr=self.opt.lr)
            self.optimizerG = optim.RMSprop(itertools.chain(self.netG.parameters(), self.netE.parameters()),
                                            lr=self.opt.lr)

        else:
            raise ValueError('Unknown optimizer: ' + self.opt.optimizer)

        # Create the schedulers
        if self.opt.lr_sched_type == 'step':
            LR_fn = optim.lr_scheduler.StepLR
        elif self.opt.lr_sched_type == 'exp':
            LR_fn = optim.lr_scheduler.ExponentialLR
        elif self.opt.lr_sched_type is None:
            LR_fn = None
        else:
            raise ValueError('Unknown scheduler')

        self.optG_z_lr_scheduler = LR_fn(
            self.optimizerG, step_size=self.opt.z_lr_sched_step,
            gamma=self.opt.z_lr_sched_gamma)
        self.LR_SCHED_MAP = [self.optG_z_lr_scheduler]
        self.OPT_MAP = [self.optimizerG]

    def get_samples(self):
        """Get samples."""
        try:
            samples = self.data_iter.next()
        except StopIteration:
            del self.data_iter
            self.data_iter = iter(self.dataset_loader)
            samples = self.data_iter.next()
        except AttributeError:
            self.data_iter = iter(self.dataset_loader)
            samples = self.data_iter.next()
        return samples

    def get_real_samples(self):
        """Get a real sample."""

        # If iterate through already rendered images
        if self.opt.img_iterate:
            data, data_depth, data_cond, light_pos1 = next(self.img_iter)
            self.light_pos1 = light_pos1
            data = [tch_var_f(d).permute(2, 0, 1)/255. for d in data]
            data_depth = [tch_var_f(d_d).unsqueeze(0)/255. for d_d in data_depth]
            data_cond = [tch_var_f(cond/8) for cond in data_cond]

        # Else, render using differentiable renderer
        else:

            data, data_depth, data_normal, data_cond, real_normals, real_depths = [], [], [], [], [], []

            # Define the camera poses
            if not self.opt.same_view:
                if self.opt.full_sphere_sampling:
                    self.cam_pos = uniform_sample_sphere(
                        radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                        axis=None, angle=self.opt.angle,
                        theta_range=np.deg2rad(self.opt.theta), phi_range=np.deg2rad(self.opt.phi))
                else:
                    self.cam_pos = uniform_sample_sphere(
                        radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                        axis=None, angle=self.opt.angle,
                        theta_range=np.deg2rad(self.opt.theta),
                        phi_range=np.deg2rad(self.opt.phi))
            if self.opt.full_sphere_sampling_light:
                self.light_pos1 = uniform_sample_sphere(
                    radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                    axis=None, angle=self.opt.angle,
                    theta_range=np.deg2rad(self.opt.theta),
                    phi_range=np.deg2rad(self.opt.phi))
                # self.light_pos2 = uniform_sample_sphere(radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                #                                      axis=self.opt.axis, angle=np.deg2rad(40),
                #                                      theta_range=self.opt.theta, phi_range=self.opt.phi)
            else:
                print("inbox")
                light_eps = 0.15
                self.light_pos1 = np.random.rand(self.opt.batchSize, 3)*self.opt.cam_dist + light_eps
                self.light_pos2 = np.random.rand(self.opt.batchSize, 3)*self.opt.cam_dist + light_eps

                # TODO: deg2rad in all the angles????

            # Create a splats rendering scene
            large_scene = create_scene(self.opt.width, self.opt.height,
                                       self.opt.fovy, self.opt.focal_length,
                                       self.opt.n_splats)
            lookat = self.opt.at if self.opt.at is not None else [0.0, 0.0, 0.0, 1.0]
            large_scene['camera']['at'] = tch_var_f(lookat)

            # Render scenes
            for idx in range(self.opt.batchSize):
                # Save the splats into the rendering scene
                if self.opt.use_mesh:
                    if 'sphere' in large_scene['objects']:
                        del large_scene['objects']['sphere']
                    if 'disk' in large_scene['objects']:
                        del large_scene['objects']['disk']
                    if 'triangle' not in large_scene['objects']:
                        large_scene['objects'] = {
                            'triangle': {'face': None, 'normal': None,
                                         'material_idx': None}}
                    samples = self.get_samples()

                    large_scene['objects']['triangle']['material_idx'] = tch_var_l(
                        np.zeros(samples['mesh']['face'][0].shape[0],
                                 dtype=int).tolist())
                    if not self.opt.no_cuda:
                        large_scene['objects']['triangle']['face'] = Variable(
                            samples['mesh']['face'][0].cuda(), requires_grad=False)
                        large_scene['objects']['triangle']['normal'] = Variable(
                            samples['mesh']['normal'][0].cuda(),
                            requires_grad=False)
                    else:
                        large_scene['objects']['triangle']['face'] = Variable(
                            samples['mesh']['face'][0], requires_grad=False)
                        large_scene['objects']['triangle']['normal'] = Variable(
                            samples['mesh']['normal'][0],
                            requires_grad=False)
                else:
                    if 'sphere' in large_scene['objects']:
                        del large_scene['objects']['sphere']
                    if 'triangle' in large_scene['objects']:
                        del large_scene['objects']['triangle']
                    if 'disk' not in large_scene['objects']:
                        large_scene['objects'] = {
                            'disk': {'pos': None,
                                     'normal': None,
                                     'material_idx': None}}
                    large_scene['objects']['disk']['radius'] = tch_var_f(
                        np.ones(self.opt.n_splats) * self.opt.splats_radius)
                    large_scene['objects']['disk']['material_idx'] = tch_var_l(
                        np.zeros(self.opt.n_splats, dtype=int).tolist())
                    if not self.opt.no_cuda:
                        large_scene['objects']['disk']['pos'] = Variable(
                            samples['splats']['pos'][idx].cuda(),
                            requires_grad=False)
                        large_scene['objects']['disk']['normal'] = Variable(
                            samples['splats']['normal'][idx].cuda(),
                        requires_grad=False)
                    else:
                        large_scene['objects']['disk']['pos'] = Variable(
                            samples['splats']['pos'][idx],
                            requires_grad=False)
                        large_scene['objects']['disk']['normal'] = Variable(
                            samples['splats']['normal'][idx],
                        requires_grad=False)

                # Set camera position
                if not self.opt.same_view:
                    large_scene['camera']['eye'] = tch_var_f(self.cam_pos[idx])
                else:
                    large_scene['camera']['eye'] = tch_var_f(self.cam_pos[0])

                large_scene['lights']['pos'][0, :3]=tch_var_f(self.light_pos1[idx])
                #large_scene['lights']['pos'][1,:3]=tch_var_f(self.light_pos2[idx])

                # Render scene
                res = render(large_scene,
                             norm_depth_image_only=self.opt.norm_depth_image_only,
                             double_sided=True, use_quartic=self.opt.use_quartic)

                # Get rendered output
                if self.opt.render_img_nc == 1:
                    depth = res['depth']
                    im_d = depth.unsqueeze(0)
                    min_v = torch.min(im_d)
                    range_v = torch.max(im_d) - min_v
                    if range_v.data.cpu().numpy()[0] > 0:
                        im_d = (im_d.clone() - min_v) / range_v
                else:
                    depth = res['depth']
                    im_d = depth.unsqueeze(0)
                    min_v = torch.min(im_d)
                    range_v = torch.max(im_d) - min_v
                    # if range_v.data.cpu().numpy()[0] > 0:
                    #     im_d = (im_d.clone() - min_v) / range_v
                    if range_v.data.cpu().numpy() > 0:
                        im_d = (im_d.clone() - min_v) / range_v
                    im = res['image'].permute(2, 0, 1)
                    target_normal_ = get_data(res['normal'])
                    target_normalmap_img_ = get_normalmap_image(target_normal_)
                    im_n = tch_var_f(
                        target_normalmap_img_).view(im.shape[1], im.shape[2],
                                                    3).permute(2, 0, 1)

                # Append images
                if self.iteration_no % self.opt.save_image_interval == 0:
                    real_normals.append(target_normalmap_img_)
                    real_depths.append(get_data(depth))

                # Add depth image to the output structure
                data.append(im)
                data_depth.append(im_d)
                data_normal.append(im_n)
                data_cond.append(large_scene['camera']['eye'])

        # import pdb; pdb.set_trace()

        # np.savez('data', data=data[0].data.cpu().numpy(), data_depth=data_depth[0].data.cpu().numpy(), data_cond=data_cond[0].data.cpu().numpy())
        # sys.exit()

        # Make images
        if not self.opt.img_iterate and self.iteration_no % self.opt.save_image_interval == 0:
            torchvision.utils.save_image(
                tch_var_f(np.array(real_normals)).data.permute(0, 3, 1, 2),
                os.path.join(self.opt.vis_images, 'real_normals_{:05d}.png'.format(self.iteration_no)), nrow=2, normalize=True, scale_each=True)
            torchvision.utils.save_image(
                tch_var_f(np.array(real_depths)).data.unsqueeze(1),
                os.path.join(self.opt.vis_images, 'real_depths_{:05d}.png'.format(self.iteration_no)), nrow=2, normalize=True, scale_each=True)

        # Stack real samples
        real_samples = torch.stack(data)
        real_samples_depth = torch.stack(data_depth)
        # real_samples_normal = torch.stack(data_normal)
        real_samples_cond = torch.stack(data_cond)
        self.batch_size = real_samples.size(0)
        if not self.opt.no_cuda:
            real_samples = real_samples.cuda()
            real_samples_depth = real_samples_depth.cuda()
            # real_samples_normal = real_samples_normal.cuda()
            real_samples_cond = real_samples_cond.cuda()

        # Set input/output variables
        self.input.resize_as_(real_samples.data).copy_(real_samples.data)
        self.input_depth.resize_as_(real_samples_depth.data).copy_(real_samples_depth.data)
        # self.input_normal.resize_as_(real_samples_normal.data).copy_(real_samples_normal.data)
        self.input_cond.resize_as_(real_samples_cond.data).copy_(real_samples_cond.data)
        self.label.resize_(self.batch_size).fill_(self.real_label)
        # TODO: Remove Variables
        self.inputv = Variable(self.input)
        self.inputv_depth = Variable(self.input_depth)
        # self.inputv_normal = Variable(self.input_normal)
        self.inputv_cond = Variable(self.input_cond)
        self.labelv = Variable(self.label)

    def generate_noise_vector(self, ):
        """Generate a noise vector."""
        self.noise.resize_(
            self.batch_size, int(self.opt.nz), 1, 1).normal_(0, 1)
        self.noisev = Variable(self.noise)  # TODO: Add volatile=True???

    def tensorboard_pos_hook(self, grad):

        self.writer.add_image("position_gradient_im",
                                torch.sqrt(torch.sum(grad ** 2, dim=-1)),

                               self.iteration_no)
        self.writer.add_scalar("position_mean_channel1",
                               get_data(torch.mean(torch.abs(grad[:,:,0]))),
                               self.iteration_no)
        self.writer.add_scalar("position_gradient_mean_channel2",
                               get_data(torch.mean(torch.abs(grad[:,:,1]))),
                               self.iteration_no)
        self.writer.add_scalar("position_gradient_mean_channel3",
                               get_data(torch.mean(torch.abs(grad[:,:,2]))),
                               self.iteration_no)
        self.writer.add_scalar("position_gradient_mean",
                               get_data(torch.mean(grad)),
                               self.iteration_no)
        self.writer.add_histogram("position_gradient_hist_channel1", grad[:,:,0].clone().cpu().data.numpy(),self.iteration_no)
        self.writer.add_histogram("position_gradient_hist_channel2", grad[:,:,1].clone().cpu().data.numpy(),self.iteration_no)
        self.writer.add_histogram("position_gradient_hist_channel3", grad[:,:,2].clone().cpu().data.numpy(),self.iteration_no)
        self.writer.add_histogram("position_gradient_hist_norm", torch.sqrt(torch.sum(grad ** 2, dim=-1)).clone().cpu().data.numpy(),self.iteration_no)
        #print('grad', grad)

    def tensorboard_normal_hook(self, grad):

        self.writer.add_image("normal_gradient_im",
                                torch.sqrt(torch.sum(grad ** 2, dim=-1)),

                               self.iteration_no)
        self.writer.add_scalar("normal_gradient_mean_channel1",
                               get_data(torch.mean(torch.abs(grad[:,:,0]))),
                               self.iteration_no)
        self.writer.add_scalar("normal_gradient_mean_channel2",
                               get_data(torch.mean(torch.abs(grad[:,:,1]))),
                               self.iteration_no)
        self.writer.add_scalar("normal_gradient_mean_channel3",
                               get_data(torch.mean(torch.abs(grad[:,:,2]))),
                               self.iteration_no)
        self.writer.add_scalar("normal_gradient_mean",
                               get_data(torch.mean(grad)),
                               self.iteration_no)
        self.writer.add_histogram("normal_gradient_hist_channel1", grad[:,:,0].clone().cpu().data.numpy(),self.iteration_no)
        self.writer.add_histogram("normal_gradient_hist_channel2", grad[:,:,1].clone().cpu().data.numpy(),self.iteration_no)
        self.writer.add_histogram("normal_gradient_hist_channel3", grad[:,:,2].clone().cpu().data.numpy(),self.iteration_no)
        self.writer.add_histogram("normal_gradient_hist_norm", torch.sqrt(torch.sum(grad ** 2, dim=-1)).clone().cpu().data.numpy(),self.iteration_no)
        #print('grad', grad)

    def tensorboard_z_hook(self, grad):

        self.writer.add_scalar("z_gradient_mean",
                               get_data(torch.mean(torch.abs(grad))),
                               self.iteration_no)
        self.writer.add_histogram("z_gradient_hist_channel", grad.clone().cpu().data.numpy(),self.iteration_no)

        self.writer.add_image("z_gradient_im",
                               grad,
                               self.iteration_no)

    def render_batch(self, batch, batch_cond=None):
        """Render a batch of splats."""
        batch_size = batch.size()[0]

        # Generate camera positions on a sphere
        if batch_cond is None:
            if self.opt.full_sphere_sampling:
                cam_pos = uniform_sample_sphere(
                    radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                    axis=self.opt.axis, angle=np.deg2rad(self.opt.angle),
                    theta_range=self.opt.theta, phi_range=self.opt.phi)
                # TODO: deg2grad!!
            else:
                cam_pos = uniform_sample_sphere(
                    radius=self.opt.cam_dist, num_samples=self.opt.batchSize,
                    axis=self.opt.axis, angle=self.opt.angle,
                    theta_range=np.deg2rad(self.opt.theta),
                    phi_range=np.deg2rad(self.opt.phi))
                # TODO: deg2grad!!

        rendered_data, rendered_data_depth, rendered_data_cond, scenes = [], [], [], []
        gen_normals, gen_depths, gen_world_normals = [], [], []
        inpath = self.opt.vis_images + '/'
        inpath_xyz = self.opt.vis_xyz + '/'
        z_min = self.scene['camera']['focal_length']
        z_max = z_min + 3

        # TODO (fmannan): Move this in init. This only needs to be done once!
        # Set splats into rendering scene
        if 'sphere' in self.scene['objects']:
            del self.scene['objects']['sphere']
        if 'triangle' in self.scene['objects']:
            del self.scene['objects']['triangle']
        if 'disk' not in self.scene['objects']:
            self.scene['objects'] = {'disk': {'pos': None, 'normal': None,
                                              'material_idx': None}}
        lookat = self.opt.at if self.opt.at is not None else [0.0, 0.0, 0.0, 1.0]
        self.scene['camera']['at'] = tch_var_f(lookat)
        self.scene['objects']['disk']['material_idx'] = tch_var_l(
            np.zeros(self.opt.splats_img_size * self.opt.splats_img_size))
        loss = 0.0
        loss_ = 0.0
        z_loss_ = 0.0
        z_norm_loss_ = 0.0
        spatial_loss_ = 0.0
        spatial_var_loss_ = 0.0
        unit_normal_loss_ = 0.0
        normal_away_from_cam_loss_ = 0.0
        image_depth_consistency_loss_ = 0.0
        for idx in range(batch_size):
            # Get splats positions and normals
            eps = 1e-3
            if self.opt.rescaled:
                z = F.relu(-batch[idx][:, 0]) + z_min
                z = ((z - z.min()) / (z.max() - z.min() + eps) *
                     (z_max - z_min) + z_min)
                pos = -z
            else:
                z = F.relu(-batch[idx][:, 0]) + z_min
                pos = -F.relu(-batch[idx][:, 0]) - z_min

            self.scene['objects']['disk']['pos'] = pos

            # Normal estimation network and est_normals don't go together
            self.scene['objects']['disk']['normal'] = None

            # Set camera position
            if batch_cond is None:
                if not self.opt.same_view:
                    self.scene['camera']['eye'] = tch_var_f(cam_pos[idx])
                else:
                    self.scene['camera']['eye'] = tch_var_f(cam_pos[0])
            else:
                if not self.opt.same_view:
                    self.scene['camera']['eye'] = batch_cond[idx]
                else:
                    self.scene['camera']['eye'] = batch_cond[0]

            self.scene['lights']['pos'][0, :3] = tch_var_f(self.light_pos1[idx])
            #self.scene['lights']['pos'][1,:3] = tch_var_f(self.light_pos2[idx])

            # Render scene
            # res = render_splats_NDC(self.scene)
            res = render_splats_along_ray(self.scene,
                                          samples=self.opt.pixel_samples,
                                          normal_estimation_method='plane')

            world_tform = cam_to_world(res['pos'].view((-1, 3)),
                                       res['normal'].view((-1, 3)),
                                       self.scene['camera'])

            # Get rendered output
            res_pos = res['pos'].contiguous()
            res_pos_2D = res_pos.view(res['image'].shape)
            # The z_loss needs to be applied after supersampling
            # TODO: Enable this (currently final loss becomes NaN!!)
            # loss += torch.mean(
            #    (10 * F.relu(z_min - torch.abs(res_pos[..., 2]))) ** 2 +
            #    (10 * F.relu(torch.abs(res_pos[..., 2]) - z_max)) ** 2)


            res_normal = res['normal']
            # depth_grad_loss = spatial_3x3(res['depth'][..., np.newaxis])
            # grad_img = grad_spatial2d(torch.mean(res['image'], dim=-1)[..., np.newaxis])
            # grad_depth_img = grad_spatial2d(res['depth'][..., np.newaxis])
            image_depth_consistency_loss = depth_rgb_gradient_consistency(
                res['image'], res['depth'])
            unit_normal_loss = unit_norm2_L2loss(res_normal, 10.0)  # TODO: MN
            normal_away_from_cam_loss = away_from_camera_penalty(
                res_pos, res_normal)
            z_pos = res_pos[..., 2]
            z_loss = torch.mean((2 * F.relu(z_min - torch.abs(z_pos))) ** 2 +
                                (2 * F.relu(torch.abs(z_pos) - z_max)) ** 2)
            z_norm_loss = normal_consistency_cost(
                res_pos, res['normal'], norm=1)
            spatial_loss = spatial_3x3(res_pos_2D)
            spatial_var = torch.mean(res_pos[..., 0].var() +
                                     res_pos[..., 1].var() +
                                     res_pos[..., 2].var())
            spatial_var_loss = (1 / (spatial_var + 1e-4))

            loss = (self.opt.zloss * z_loss +
                    self.opt.unit_normalloss*unit_normal_loss +
                    self.opt.normal_consistency_loss_weight * z_norm_loss +
                    self.opt.spatial_var_loss_weight * spatial_var_loss +
                    self.opt.grad_img_depth_loss*image_depth_consistency_loss +
                    self.opt.spatial_loss_weight * spatial_loss)
            pos_out_ = get_data(res['pos'])
            loss_ += get_data(loss)
            z_loss_ += get_data(z_loss)
            z_norm_loss_ += get_data(z_norm_loss)
            spatial_loss_ += get_data(spatial_loss)
            spatial_var_loss_ += get_data(spatial_var_loss)
            unit_normal_loss_ += get_data(unit_normal_loss)
            normal_away_from_cam_loss_ += get_data(normal_away_from_cam_loss)
            image_depth_consistency_loss_ += get_data(
                image_depth_consistency_loss)
            normals_ = get_data(res_normal)

            if self.opt.render_img_nc == 1:
                depth = res['depth']
                im = depth.unsqueeze(0)
            else:
                depth = res['depth']
                im_d = depth.unsqueeze(0)
                im = res['image'].permute(2, 0, 1)
                H, W = im.shape[1:]
                target_normal_ = get_data(res['normal']).reshape((H, W, 3))
                target_normalmap_img_ = get_normalmap_image(target_normal_)
                target_worldnormal_ = get_data(world_tform['normal']).reshape(
                    (H, W, 3))
                target_worldnormalmap_img_ = get_normalmap_image(
                    target_worldnormal_)

            if self.iteration_no % self.opt.save_image_interval == 0:
                gen_normals.append(target_normalmap_img_)
                gen_depths.append(get_data(res['depth']))
                gen_world_normals.append(target_worldnormalmap_img_)

            if self.iteration_no % 200 == 0:
                im2 = get_data(res['image'])
                depth2 = get_data(res['depth'])
                pos = get_data(res['pos'])

                out_file2 = ("pos"+".npy")
                np.save(inpath_xyz+out_file2, pos)

                out_file2 = ("im"+".npy")
                np.save(inpath_xyz+out_file2, im2)

                out_file2 = ("depth"+".npy")
                np.save(inpath_xyz+out_file2, depth2)

                # Save xyz file
                save_xyz((inpath_xyz + str(self.iteration_no) +
                          'withnormal_{:05d}.xyz'.format(idx)),
                         pos=get_data(res['pos']),
                         normal=get_data(res['normal']))

                # Save xyz file in world coordinates
                save_xyz((inpath_xyz + str(self.iteration_no) +
                          'withnormal_world_{:05d}.xyz'.format(idx)),
                         pos=get_data(world_tform['pos']),
                         normal=get_data(world_tform['normal']))
            if self.opt.gz_gi_loss is not None and self.opt.gz_gi_loss > 0:
                gradZ = grad_spatial2d(res_pos_2D[:, :, 2][:, :, np.newaxis])
                gradImg = grad_spatial2d(torch.mean(im,
                                                    dim=0)[:, :, np.newaxis])
                for (gZ, gI) in zip(gradZ, gradImg):
                    loss += (self.opt.gz_gi_loss * torch.mean(torch.abs(
                                torch.abs(gZ) - torch.abs(gI))))

            # Normalize rendered image
            im = im/torch.max(im)*255.

            # Store normalized depth into the data
            rendered_data.append(im)
            rendered_data_depth.append(im_d)
            rendered_data_cond.append(self.scene['camera']['eye'])
            scenes.append(self.scene)

        # Save batch of images
        if self.iteration_no % self.opt.save_image_interval == 0:
            torchvision.utils.save_image(
                tch_var_f(np.array(gen_normals)).data.permute(0, 3, 1, 2),
                os.path.join(self.opt.vis_images, 'gen_normals_{:05d}.png'.format(self.iteration_no)), nrow=2, normalize=True, scale_each=True)
            torchvision.utils.save_image(
                tch_var_f(np.array(gen_depths)).data.unsqueeze(1),
                os.path.join(self.opt.vis_images, 'gen_depths_{:05d}.png'.format(self.iteration_no)), nrow=2, normalize=True, scale_each=True)
            torchvision.utils.save_image(
                tch_var_f(np.array(gen_world_normals)).data.permute(0, 3, 1, 2),
                os.path.join(self.opt.vis_images, 'gen_world_normals_{:05d}.png'.format(self.iteration_no)), nrow=2, normalize=True, scale_each=True)

        rendered_data = torch.stack(rendered_data)
        rendered_data_depth = torch.stack(rendered_data_depth)

        if self.iteration_no % self.opt.print_interval*4 == 0 and self.in_critic == 0:
            z__ = pos_out_[..., 2]
            self.writer.add_scalar("loss",
                                   loss_/self.opt.batchSize,
                                   self.iteration_no)
            self.writer.add_scalar("nloss",
                                   unit_normal_loss_/self.opt.batchSize,
                                   self.iteration_no)
            self.writer.add_scalar("minz",
                                   np.min(z__),
                                   self.iteration_no)
            self.writer.add_scalar("maxz",
                                   np.max(z__),
                                   self.iteration_no)
            self.writer.add_scalar("min_normal",
                                   normals_[..., 2].min(),
                                   self.iteration_no)
            self.writer.add_scalar("max_normal",
                                   normals_[..., 2].max(),
                                   self.iteration_no)
            self.writer.add_scalar("z_loss",
                                   z_loss_/self.opt.batchSize,
                                   self.iteration_no)
            self.writer.add_scalar("z_normal_loss",
                                   z_norm_loss_/self.opt.batchSize,
                                   self.iteration_no)
            self.writer.add_scalar("spatial_var_loss",
                                   spatial_var_loss_/self.opt.batchSize,
                                   self.iteration_no)
            self.writer.add_scalar("normal_away_loss",
                                   normal_away_from_cam_loss_/self.opt.batchSize,
                                   self.iteration_no)
            self.writer.add_scalar("spatial_loss",
                                   spatial_loss_/self.opt.batchSize,
                                   self.iteration_no)
            self.writer.add_scalar("im_depth_cons_loss",
                                   image_depth_consistency_loss_/self.opt.batchSize,
                                   self.iteration_no)

            res['pos'].register_hook(self.tensorboard_pos_hook)
            res['normal'].register_hook(self.tensorboard_normal_hook)

            log_to_print = ('it: %d, loss: %f, nloss: %f, z_loss:%f, [%f, %f], '
                            'z_normal_loss: %f, spatial_var_loss: %f, '
                            'normal_away_loss: %f, nz_range: [%f, %f], '
                            'spatial_loss: %f, im_depth_cons_loss: %f' %
                            (self.iteration_no,
                             loss_/self.opt.batchSize,
                             unit_normal_loss_/self.opt.batchSize,
                             z_loss_/self.opt.batchSize,
                             np.min(z__), np.max(z__),
                             z_norm_loss_/self.opt.batchSize,
                             spatial_var_loss_/self.opt.batchSize,
                             normal_away_from_cam_loss_/self.opt.batchSize,
                             normals_[..., 2].min(), normals_[..., 2].max(),
                             spatial_loss_/self.opt.batchSize,
                             image_depth_consistency_loss_/self.opt.batchSize))
            print(log_to_print)
            self.output_loss_file.write(log_to_print)
            self.output_loss_file.flush()
        return rendered_data, rendered_data_depth, loss/self.opt.batchSize

    def tensorboard_hook(self, grad):
        self.writer.add_scalar("z_gradient_mean",
                               get_data(torch.mean(grad[0])),
                               self.iteration_no)
        self.writer.add_histogram("z_gradient_hist_channel", grad[0].clone().cpu().data.numpy(),self.iteration_no)

        self.writer.add_image("z_gradient_im",
                               grad[0].view(self.opt.splats_img_size,self.opt.splats_img_size),
                               self.iteration_no)

    def train(self):
        """Train network."""

        # Load pretrained model if required
        if self.opt.gen_model_path is not None:
            print("Reloading networks from")
            print(' > Generator', self.opt.gen_model_path)
            self.netG.load_state_dict(
                torch.load(open(self.opt.gen_model_path, 'rb')))
            print(' > Generator2', self.opt.gen_model_path2)
            self.netG2.load_state_dict(
                torch.load(open(self.opt.gen_model_path2, 'rb')))
            print(' > Discriminator', self.opt.dis_model_path)
            self.netD.load_state_dict(
                torch.load(open(self.opt.dis_model_path, 'rb')))
            print(' > Discriminator2', self.opt.dis_model_path2)
            self.netD2.load_state_dict(
                torch.load(open(self.opt.dis_model_path2, 'rb')))

        # Load image & cam_pos iterator if required
        if self.opt.img_iterate:
            self.img_iter = Iterator(root_dir=self.opt.root_dir, batch_size=self.opt.batchSize, shuffle=True)

        # Arrays
        self.G_losses = []
        self.D_losses = []
        self.E_losses = []
        self.reconstruction_losses = []
        self.wass_Ds = []
        self.G_grad_norms = []
        self.D_grad_norms = []
        self.D_xs = []
        self.D_Gz_trainDs = []
        self.D_Gz_trainGs = []
        self.D_x_reczs = []

        # Start training
        start_time = time.time()
        file_name = os.path.join(self.opt.out_dir, 'log.txt')
        with open(file_name, 'wt') as log_file:

            curr_generator_idx = 0

            for iteration in range(self.opt.n_iter):

                self.iteration_no = iteration

                ############################
                # (1) Update D network
                ############################

                # Train Discriminator critic_iters times
                for j in range(self.opt.critic_iters):

                    self.in_critic = 1
                    self.netD.zero_grad()

                    # Train D with real
                    #################

                    # Get real samples:
                    # inputv, inputv_depth, inputv_normal, inputv_cond, labelv
                    self.get_real_samples()

                    # Get z from x (for ALI)
                    # z <--GaussReparam-- mu, var <--E-- x
                    mu_z, logvar_z = self.netE(self.inputv)
                    z_real = gauss_reparametrize(mu_z, logvar_z)

                    # Get discriminator output
                    # input_D = torch.cat([self.inputv, self.inputv_depth], 1)
                    outD_real = self.netD(self.inputv, self.inputv_cond.detach(), z_real.detach())

                    # Backprop thru D
                    if self.opt.criterion == 'GAN':
                        errD_real = self.criterion(outD_real, self.labelv)
                        errD_real.backward()
                    elif self.opt.criterion == 'WGAN':
                        errD_real = outD_real.mean()
                        errD_real.backward(self.mone)
                    else:
                        raise ValueError('Unknown GAN criterium')

                    # Train D with fake
                    #################

                    # Render fake data
                    # noise --G--> z --render--> x
                    self.generate_noise_vector()
                    fake_z = self.netG(self.noisev, self.inputv_cond)
                    # The normal generator is dependent on z
                    fake_rendered, fd, loss = self.render_batch(
                        fake_z, self.inputv_cond)

                    # Get discriminator output
                    # Do not bp through G
                    outD_fake = self.netD(fake_rendered.detach(),
                                          self.inputv_cond.detach(), self.noisev.detach())

                    # Backprop thru D
                    if self.opt.criterion == 'GAN':
                        labelv = Variable(self.label.fill_(self.fake_label))
                        errD_fake = self.criterion(outD_fake, labelv)
                        errD_fake.backward()
                        errD = errD_real + errD_fake
                    elif self.opt.criterion == 'WGAN':
                        errD_fake = outD_fake.mean()
                        errD_fake.backward(self.one)
                        errD = errD_fake - errD_real
                    else:
                        raise ValueError('Unknown GAN criterium')

                    # Compute gradient penalty, and backprop thru D
                    if self.opt.gp != 'None':
                        gradient_penalty = calc_gradient_penalty(
                            self.netD, self.netE, self.inputv.data, fake_rendered.data,
                            self.inputv_cond.data, self.noisev.data, z_real.data, self.opt.gp_lambda, self.opt.no_cuda)
                        gradient_penalty.backward()
                        errD += gradient_penalty

                    # Perform gradient clipping
                    gnorm_D = torch.nn.utils.clip_grad_norm(
                        self.netD.parameters(), self.opt.max_gnorm)  # TODO

                    # Update weights of D
                    self.optimizerD.step()

                    # Clamp critic weigths if not GP and if WGAN
                    if self.opt.criterion == 'WGAN' and self.opt.gp == 'None':
                        for p in self.netD.parameters():
                            p.data.clamp_(-self.opt.clamp, self.opt.clamp)

                ############################
                # (2) Update G network
                ############################

                self.in_critic = 0
                self.netG.zero_grad()
                self.netE.zero_grad()

                # To avoid computation
                # for p in self.netD.parameters():
                #     p.requires_grad = False

                # Render fake samples
                # noise --G--> z --render--> x
                self.generate_noise_vector()
                fake_z = self.netG(self.noisev, self.inputv_cond)
                fake_rendered, fd, loss = self.render_batch(
                    fake_z, self.inputv_cond)

                if iteration % self.opt.print_interval*4 == 0:
                    fake_z.register_hook(self.tensorboard_hook)

                # Get discriminator output for fake
                outG_fake = self.netD(fake_rendered, self.inputv_cond, self.noisev)

                # Backprop thru G
                if self.opt.criterion == 'GAN':
                    # Fake labels are real for generator cost
                    labelv = Variable(self.label.fill_(self.real_label))
                    errG = self.criterion(outG_fake, labelv)
                    errG.backward()
                elif self.opt.criterion == 'WGAN':
                    errG = outG_fake.mean() + loss
                    errG.backward(self.mone, retain_graph=True)
                else:
                    raise ValueError('Unknown GAN criterium')

                # Get z from x (for ALI)
                mu_z, logvar_z = self.netE(self.inputv)
                z_real = gauss_reparametrize(mu_z, logvar_z)

                # Get discriminator output for real, with require_grad thru G True
                # input_D = torch.cat([self.inputv, self.inputv_depth], 1)
                real_output_z = self.netD(self.inputv, self.inputv_cond, z_real)

                # Backprop thru G
                if self.opt.criterion == 'GAN':
                    errE = self.criterion(real_output_z, self.labelv)
                    errE.backward()
                elif self.opt.criterion == 'WGAN':
                    errE = real_output_z.mean()
                    errE.backward(self.one, retain_graph=True)
                else:
                    raise ValueError('Unknown GAN criterium')

                # To check how G is, render some samples using z_real
                # noise --G--> z --render--> x
                reconstruction_z = self.netG(z_real, self.inputv_cond)
                reconstruction_rendered, reconstructiond, loss = self.render_batch(
                    reconstruction_z, self.inputv_cond)

                # Backprop thru G from reconstruction loss
                if not self.opt.no_cuda:
                    mse_criterion = nn.MSELoss().cuda()
                else:
                    mse_criterion = nn.MSELoss()
                reconstruction_loss = mse_criterion(reconstruction_rendered, self.inputv)
                reconstruction_loss.backward()

                # Clip gradients
                gnorm_G = torch.nn.utils.clip_grad_norm(
                    self.netG.parameters(), self.opt.max_gnorm)  # TODO

                # Update weights of G
                # along with some scheduling
                if (self.opt.alt_opt_zn_interval is not None and
                    iteration >= self.opt.alt_opt_zn_start):
                    # update one of the generators
                    if (((iteration - self.opt.alt_opt_zn_start) %
                         self.opt.alt_opt_zn_interval) == 0):
                        # switch generator vars to optimize
                        curr_generator_idx = (1 - curr_generator_idx)
                    if iteration < self.opt.lr_iter:
                        self.LR_SCHED_MAP[curr_generator_idx].step()
                        self.OPT_MAP[curr_generator_idx].step()
                else:
                    if iteration < self.opt.lr_iter:
                        self.optG_z_lr_scheduler.step()
                    # Update weights of G
                    self.optimizerG.step()

                # Log print
                if iteration % self.opt.print_interval == 0:

                    l2_loss = mse_criterion(fd, self.inputv_depth)
                    Wassertein_D = (errD_real.data[0] - errD_fake.data[0])
                    self.writer.add_scalar("Loss_G",
                                           errG.data[0],
                                           self.iteration_no)
                    self.writer.add_scalar("Loss_D",
                                           errD.data[0],
                                           self.iteration_no)
                    self.writer.add_scalar("Loss_E",
                                           errE.data[0],
                                           self.iteration_no)
                    self.writer.add_scalar("Reconstruction_Loss",
                                           reconstruction_loss.data[0],
                                           self.iteration_no)
                    self.writer.add_scalar("Wassertein_D",
                                           Wassertein_D,
                                           self.iteration_no)
                    self.writer.add_scalar("Disc_grad_norm",
                                           gnorm_D,
                                           self.iteration_no)
                    self.writer.add_scalar("Gen_grad_norm",
                                           gnorm_G,
                                           self.iteration_no)

                    curr_time = time.time()
                    curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
                    elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
                    log = '\n[{}]: Elapsed [{}] '.format(curr_time_str, elapsed) +\
                          '[%d/%d] Loss_D: %.4f, Loss_G:, %.4f, Loss_E: %.4f, reconstruction_loss: %.4f, Loss_D_real: %.4f, '\
                          'Loss_D_fake: %.4f, Wassertein_D: %.4f, '\
                          'L2_loss: %.4f, z_lr: %.8f, Disc_grad_norm: %.8f, Gen_grad_norm: %.8f\n' % (
                          iteration, self.opt.n_iter, errD.data[0], errG.data[0], errE.data[0], reconstruction_loss.data[0], errD_real.data[0],
                          errD_fake.data[0], Wassertein_D,
                          loss.data[0], self.optG_z_lr_scheduler.get_lr()[0], gnorm_D, gnorm_G)
                    print(log)
                    log_file.write(log)
                    log_file.flush()
                    mem_check()

                    # Accumulate losses and plot
                    self.G_losses.append(errG.data[0])
                    self.D_losses.append(errD.data[0])
                    self.E_losses.append(errE.data[0])
                    self.reconstruction_losses.append(reconstruction_loss.data[0])
                    self.D_xs.append(outD_real.mean().data[0])
                    self.D_Gz_trainDs.append(outD_fake.mean().data[0])
                    self.D_Gz_trainGs.append(outG_fake.mean().data[0])
                    self.D_x_reczs.append(real_output_z.mean().data[0])
                    self.G_grad_norms.append(gnorm_G)
                    self.D_grad_norms.append(gnorm_D)
                    self.wass_Ds.append(Wassertein_D)
                    self.make_plots()

                # Save output images
                if iteration % (self.opt.save_image_interval) == 0:

                    if iteration % (2*self.opt.save_image_interval) !=0:
                        # cs = tch_var_f(contrast_stretch_percentile(
                        #     get_data(fd), 200, [fd.data.min(), fd.data.max()]))
                        torchvision.utils.save_image(
                            fake_rendered.data,
                            os.path.join(self.opt.vis_images, 'gen_image_{:05d}.png'.format(iteration)), nrow=2, normalize=True, scale_each=True)
                        torchvision.utils.save_image(
                            reconstruction_rendered.data,
                            os.path.join(self.opt.vis_images, 'reconstructed_image_{:05d}.png'.format(iteration)), nrow=2, normalize=True, scale_each=True)
                    else:
                        cam_pos=self.inputv_cond[0].repeat(self.opt.batchSize,1)
                        fake_z = self.netG(self.noisev, cam_pos)
                        # The normal generator is dependent on z
                        fake_rendered, fd, loss = self.render_batch(
                            fake_z, cam_pos)
                        # cs = tch_var_f(contrast_stretch_percentile(
                        #     get_data(fd), 200, [fd.data.min(), fd.data.max()]))
                        torchvision.utils.save_image(
                            fake_rendered.data,
                            os.path.join(self.opt.vis_images, 'gen_image_{:05d}.png'.format(iteration)),
                            nrow=2, normalize=True, scale_each=True)

                # Save input images
                if iteration % (self.opt.save_image_interval) == 0:
                    # cs = tch_var_f(contrast_stretch_percentile(
                    #     get_data(fd), 200, [fd.data.min(), fd.data.max()]))
                    torchvision.utils.save_image(
                        self.inputv.data,
                        os.path.join(self.opt.vis_images, 'input_{:05d}.png'.format(iteration)), nrow=2, normalize=True, scale_each=True)

                # Do checkpointing
                if iteration % self.opt.save_interval == 0:
                    self.save_networks(iteration)

    def make_plots(self):
        iters = np.arange(len(self.G_losses))*self.opt.print_interval
        fig = plt.figure(figsize=(20, 25))
        # Losses
        plt.subplot(511)
        plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
        plt.plot(iters, self.G_losses, alpha=0.7, label='G_loss')
        plt.plot(iters, self.D_losses, alpha=0.7, label='D_loss')
        plt.plot(iters, self.E_losses, alpha=0.7, label='E_loss')
        plt.legend()
        plt.yscale("symlog")
        plt.title("Losses")
        plt.xlabel("Iterations")
        # Reconstruction Loss
        plt.subplot(512)
        plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
        plt.plot(iters, self.reconstruction_losses, label='recon_loss')
        plt.legend()
        plt.yscale("symlog")
        plt.title("Reconstruction Loss")
        plt.xlabel("Iterations")
        # D outputs
        plt.subplot(513)
        plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
        plt.plot(iters, np.ones(iters.shape), 'k--', alpha=0.5)
        plt.plot(iters, self.D_xs, alpha=0.7, label='D(x)')
        plt.plot(iters, self.D_Gz_trainDs, alpha=0.7, label='D(G(z))_trainD')
        plt.plot(iters, self.D_Gz_trainGs, alpha=0.7, label='D(G(z))_trainG')
        plt.plot(iters, self.D_x_reczs, alpha=0.7, label='D(G(rec_z))')
        plt.legend()
        plt.yscale("symlog")
        plt.title("D(x), D(G(z))")
        plt.xlabel("Iterations")
        # Gradient Norms
        plt.subplot(514)
        plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
        plt.plot(iters, self.G_grad_norms, alpha=0.7, label='G_grad_norms')
        plt.plot(iters, self.D_grad_norms, alpha=0.7, label='D_grad_norms')
        plt.legend()
        plt.yscale("symlog")
        plt.title("Graident norms")
        plt.xlabel("Iterations")
        # Gradient Norms
        plt.subplot(515)
        plt.plot(iters, np.zeros(iters.shape), 'k--', alpha=0.5)
        plt.plot(iters, self.wass_Ds, alpha=0.7, label='Wassertein_D')
        plt.legend()
        plt.yscale("symlog")
        plt.title("Wassertein_D")
        plt.xlabel("Iterations")
        # Save
        plt.savefig(os.path.join(self.opt.out_dir, "plots.png"))
        plt.clf()
        plt.close()

    def save_networks(self, epoch):
        """Save networks to hard disk."""
        torch.save(self.netG.state_dict(),
                   '%s/netG_epoch_%d.pth' % (self.opt.out_dir, epoch))
        torch.save(self.netD.state_dict(),
                   '%s/netD_epoch_%d.pth' % (self.opt.out_dir, epoch))
        torch.save(self.netE.state_dict(),
                   '%s/netE_epoch_%d.pth' % (self.opt.out_dir, epoch))
        torch.save(self.netD2.state_dict(),
                   '%s/netD2_epoch_%d.pth' % (self.opt.out_dir, epoch))

    def save_images(self, epoch, input, output):
        """Save images."""
        if self.opt.render_img_nc == 1:
            imsave(self.opt.out_dir + '/input2' + str(epoch) + '.png',
                   np.uint8(255. * input.cpu().data.numpy().squeeze()))
            imsave(self.opt.out_dir + '/fz' + str(epoch) + '.png',
                   np.uint8(255. * output.cpu().data.numpy().squeeze()))
        else:
            imsave(self.opt.out_dir + '/input2' + str(epoch) + '.png',
                   np.uint8(255. * input.cpu().data.numpy().transpose((1, 2, 0))))
            imsave(self.opt.out_dir + '/output2' + str(epoch) + '.png',
                   np.uint8(255. * output.cpu().data.numpy().transpose((1, 2, 0))))


def main():
    """Start training."""
    # Parse args
    opt = Parameters().parse()

    opt.name = "{0:%Y%m%d_%H%M%S}_{1}_{2}".format(datetime.datetime.now(), opt.name, os.path.basename(opt.root_dir.rstrip('/')))

    # Create experiment output folder
    exp_dir = os.path.join(opt.out_dir, opt.name)
    mkdirs(exp_dir)
    sub_dirs=['vis_images','vis_xyz','vis_monitoring']
    for sub_dir in sub_dirs:
        dir_path = os.path.join(exp_dir, sub_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        setattr(opt, sub_dir, dir_path)


    # Copy scripts to experiment folder
    copy_scripts_to_folder(exp_dir)

    # Save parameters to experiment folder
    file_name = os.path.join(exp_dir, 'opt.txt')
    args = vars(opt)
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    # Create dataset loader
    dataset_load = Dataset_load(opt)

    # Create GAN
    gan = GAN(opt, dataset_load, exp_dir)

    # Train gan
    gan.train()


if __name__ == '__main__':
    main()
