"""Networks."""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import functools
from diffrend.torch.GAN.generator_networks import _netG_mlp, _netG, DCGAN_G, DCGAN_G2
from diffrend.torch.GAN.discriminator_networks import _netD, _netD_256, DCGAN_D

def create_networks(opt, verbose=True, **params):
    """Create the networks."""
    # Parameters
    ngpu = int(opt.ngpu)

    # Generator parameters
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    gen_norm = _select_norm(opt.gen_norm)
    gen_nextra_layers = int(opt.gen_nextra_layers)

    # Discriminator parameters
    ndf = int(opt.ndf)
    nef = int(opt.nef)
    disc_norm = _select_norm(opt.disc_norm)
    disc_nextra_layers = int(opt.disc_nextra_layers)

    # Rendering parameters
    render_img_nc = int(opt.render_img_nc)
    splats_img_size = int(opt.splats_img_size)
    n_splats = int(opt.n_splats)
    render_img_size = int(opt.width)

    if opt.no_renderer:
        splats_n_dims = 3 # RGB channels
    else:
        if opt.fix_splat_pos:
            splats_n_dims = 1
        else:
            splats_n_dims = 3
        if opt.norm_sph_coord:
            splats_n_dims += 2
        else:
            splats_n_dims += 3

    cond_size = 6 if opt.no_renderer else 3

    # Create generator network
    if opt.gen_type == 'mlp':
        netG = _netG_mlp(ngpu, nz, ngf, splats_n_dims, n_splats)
    elif opt.gen_type == 'cnn':
        netG = _netG(ngpu, nz, ngf, splats_n_dims, use_tanh=False,
                     bias_type=opt.gen_bias_type)
    elif opt.gen_type == 'dcgan':
        netG = DCGAN_G(splats_img_size, nz, splats_n_dims, ngf, ngpu, cond_size=cond_size,
                       n_extra_layers=gen_nextra_layers, use_tanh=False,
                       norm=gen_norm)
    elif opt.gen_type == 'resnet':
        netG = _netG_resnet(nz, splats_n_dims, n_splats)
    else:
        raise ValueError("Unknown generator")

    # Init weights/load pretrained model
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    else:
        netG.apply(weights_init)

    # If WGAN not use no_sigmoid
    use_sigmoid = opt.criterion != 'WGAN'

    # Create the discriminator network
    if opt.disc_type == 'cnn':
        if render_img_size == 128:
            netD = _netD(ngpu, 3, ndf, render_img_size,nz, cond_size=cond_size,
                         use_sigmoid=use_sigmoid)
        else:
            netD = _netD_256(ngpu, 3, ndf, render_img_size,nz,
                             use_sigmoid=use_sigmoid)
        # else:
        #     netD = _netD_64(ngpu, 3, ndf, render_img_size,
        #                     use_sigmoid=use_sigmoid)
    elif opt.disc_type == 'dcgan':
        netD = DCGAN_D(render_img_size, nz, render_img_nc, ndf, ngpu,
                       n_extra_layers=disc_nextra_layers,
                       use_sigmoid=use_sigmoid, norm=disc_norm)
    else:
        raise ValueError("Unknown discriminator")


    # Init weights/load pretrained model
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    else:
        netD.apply(weights_init)

    # Show networks
    netE = LatentEncoder( nz, 3, nef)
    if opt.netE != '':
        netE.load_state_dict(torch.load(opt.netE))
    else:
        netE.apply(weights_init)
    if verbose:
        print(netG)
        print(netE)
        print(netD)

    return netG, netD, netE


def weights_init(m):
    """Weight initializer."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _select_norm(norm):
    """Select the normalization method."""
    if norm == 'batchnorm':
        norm = nn.BatchNorm2d
    elif norm == 'instancenorm':
        norm = nn.InstanceNorm2d
    elif norm == 'None' or norm is None:
        norm = None
    else:
        raise ValueError("Unknown normalization")
    return norm




class LatentEncoder(nn.Module):
    def __init__(self, nlatent, input_nc, nef):
        super(LatentEncoder, self).__init__()

        use_bias = False
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        kw = 3
        # sequence = [
        #     nn.Conv2d(input_nc, nef, kernel_size=kw, stride=2, padding=1, bias=True),
        #     nn.ReLU(True),
        #
        #     nn.Conv2d(nef, 2*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
        #     norm_layer(2*nef),
        #     nn.ReLU(True),
        #
        #     nn.Conv2d(2*nef, 4*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
        #     norm_layer(4*nef),
        #     nn.ReLU(True),
        #
        #     nn.Conv2d(4*nef, 8*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
        #     norm_layer(8*nef),
        #     nn.ReLU(True),
        #
        #     # nn.Conv2d(8*nef, 8*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
        #     # norm_layer(8*nef),
        #     # nn.ReLU(True),
        #
        #     nn.Conv2d(8*nef, 8*nef, kernel_size=4, stride=1, padding=0, bias=use_bias),
        #     norm_layer(8*nef),
        #     nn.ReLU(True),
        #
        # ]
        sequence = [
            nn.Conv2d(input_nc, nef, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(True),

            nn.Conv2d(nef, 2*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2*nef),
            nn.ReLU(True),

            nn.Conv2d(2*nef, 4*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4*nef),
            nn.ReLU(True),

            nn.Conv2d(4*nef, 4*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4*nef),
            nn.ReLU(True),

            nn.Conv2d(4*nef, 8*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(8*nef),
            nn.ReLU(True),
        nn.Conv2d(8*nef, 8*nef, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(8*nef),
            nn.ReLU(True),

            nn.Conv2d(8*nef, 8*nef, kernel_size=4, stride=1, padding=0, bias=use_bias),
            norm_layer(8*nef),
            nn.ReLU(True),
        ]

        self.conv_modules = nn.Sequential(*sequence)

        # make sure we return mu and logvar for latent code normal distribution
        self.enc_mu = nn.Conv2d(8*nef, nlatent, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc_logvar = nn.Conv2d(8*nef, nlatent, kernel_size=1, stride=1, padding=0, bias=True)

        # NOTE (Zihang): here is a difference compared to the TF code. In the TF code, there are two
        # more layers used to generate the gaussian parameters, namely (1) an instance norm layer, and
        # (2) a ReLU layer. I personally don't think there is any good reason that we should use the
        # IN or ReLU here. But maybe it's worthy running a comparison experiment later if there is a
        # significant difference.

    def forward(self, input):
        conv_out = self.conv_modules(input)
        mu = self.enc_mu(conv_out)
        logvar = self.enc_logvar(conv_out)
        return (mu.view(mu.size(0), -1), logvar.view(logvar.size(0), -1))

