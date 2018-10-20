"""Networks."""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import functools

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
    if opt.fix_splat_pos:
        splats_n_dims = 1
    else:
        splats_n_dims = 3
    if opt.norm_sph_coord:
        splats_n_dims += 2
    else:
        splats_n_dims += 3

    # Create generator network
    if opt.gen_type == 'mlp':
        netG = _netG_mlp(ngpu, nz, ngf, splats_n_dims, n_splats)
    elif opt.gen_type == 'cnn':
        netG = _netG(ngpu, nz, ngf, splats_n_dims, use_tanh=False,
                     bias_type=opt.gen_bias_type)
    elif opt.gen_type == 'dcgan':
        netG = DCGAN_G(splats_img_size, nz, splats_n_dims, ngf, ngpu,
                       n_extra_layers=gen_nextra_layers, use_tanh=False,
                       norm=gen_norm)
        if 'depth_only' in params and params['depth_only'] is True:
            netG2 = None
        else:
            netG2 = DCGAN_G2(splats_img_size, nz, splats_n_dims, ngf, ngpu,
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
    if opt.netG2 != '' and netG2 is not None:
        netG2.load_state_dict(torch.load(opt.netG2))
    elif netG2 is not None:
        netG2.apply(weights_init)

    # If WGAN not use no_sigmoid
    if opt.criterion == 'WGAN':
        use_sigmoid = False
    else:
        use_sigmoid = True

    # Create the discriminator network
    if opt.disc_type == 'cnn':
        if render_img_size == 128:
            netD = _netD(ngpu, 3, ndf, render_img_size,nz,
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

    # new network
    if opt.disc_type == 'cnn':
        if render_img_size == 128:
            netD2 = _netD(ngpu, 3, ndf, render_img_size,nz,
                          use_sigmoid=use_sigmoid)
        else:
            netD2 = _netD_256(ngpu, 3, ndf, render_img_size,nz,
                              use_sigmoid=use_sigmoid)
        # else:
        #     netD2 = _netD_64(ngpu, 3, ndf, render_img_size,
        #                      use_sigmoid=use_sigmoid)
    elif opt.disc_type == 'dcgan':
        netD2 = DCGAN_D(render_img_size, nz, render_img_nc, ndf, ngpu,
                        n_extra_layers=disc_nextra_layers,
                        use_sigmoid=use_sigmoid, norm=disc_norm)
    else:
        raise ValueError("Unknown discriminator")

    # Init weights/load pretrained model
    if opt.netD != '':
        netD2.load_state_dict(torch.load(opt.netD2))
    else:
        netD2.apply(weights_init)
    # Show networks
    netE = LatentEncoder( nz, 3, nef)
    if opt.netE != '':
        netE.load_state_dict(torch.load(opt.netE))
    else:
        netE.apply(weights_init)
    if verbose:
        print(netG)
        print(netE)
        print(netG2)
        print(netD)
        print(netD2)

    return netG, netG2, netD, netD2, netE


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


#############################################
# Modules for conditional batchnorm
#############################################
class TwoInputModule(nn.Module):
    """Abstract class."""

    def forward(self, input1, input2):
        """Forward method."""
        raise NotImplementedError


class CondBatchNorm(nn.BatchNorm2d, TwoInputModule):
    """Conditional batch norm."""

    def __init__(self, x_dim, z_dim, eps=1e-5, momentum=0.1):
        """Constructor.

        - `x_dim`: dimensionality of x input
        - `z_dim`: dimensionality of z latents
        """
        super(CondBatchNorm, self).__init__(x_dim, eps, momentum, affine=False)
        self.eps = eps
        self.shift_conv = nn.Sequential(
            nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
            # nn.ReLU(True)
        )
        self.scale_conv = nn.Sequential(
            nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
            # nn.ReLU(True)
        )

    def forward(self, input, noise):
        """Forward method."""
        shift = self.shift_conv.forward(noise)
        scale = self.scale_conv.forward(noise)

        norm_features = super(CondBatchNorm, self).forward(input)
        output = norm_features * scale + shift
        return output


#############################################
# Generators
#############################################

class _netG_mlp(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, nsplats):
        super(_netG_mlp, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nspalts = nsplats
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz, ngf * 4),
            # nn.BatchNorm1d(ngf*4),
            nn.LeakyReLU(0.2),

            nn.Linear(ngf * 4, ngf * 16),
            nn.BatchNorm1d(ngf * 16),
            nn.LeakyReLU(0.2),

            nn.Linear(ngf * 16, ngf * 16),
            nn.BatchNorm1d(ngf * 16),
            nn.LeakyReLU(0.2),

            nn.Linear(ngf * 16, ngf * 32),
            # nn.BatchNorm1d(ngf*16),
            nn.LeakyReLU(0.2),

            nn.Linear(ngf * 32, ngf * 64),
            nn.LeakyReLU(0.2),

            nn.Linear(ngf * 64, nc * nsplats)
            # nn.BatchNorm1d(ndf*4),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(ndf*4, 1)
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            input = input.view(input.size()[0], input.size()[1])
            output = self.main(input)
            output = output.view(output.size(0), self.nspalts, self.nc)
        return output


class ReshapeSplats(nn.Module):
    """Reshape the splats from a 2D to a 1D shape."""

    def forward(self, x):
        """Forward method."""
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        return x


class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, use_tanh=False, bias_type=None):
        super(_netG, self).__init__()
        # Save parameters
        self.ngpu = ngpu
        self.nc = nc
        self.use_tanh = use_tanh
        self.bias_type = bias_type

        # Main layers
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, nc, 4, 2, 1, bias=False),
            # state size. (ngf) x 32 x 32
        )

        self.reshape = ReshapeSplats()

        # Coordinates bias
        if bias_type == 'plane':
            coords_tmp = np.array(list(np.ndindex((32, 32)))).reshape(2, 32,
                                                                      32)
            coords = np.zeros((1, nc, 32, 32), dtype=np.float32)
            coords[0, :2, :, :] = coords_tmp / 32.
            self.coords = Variable(torch.FloatTensor(coords))
            if torch.cuda.is_available():
                self.coords = self.coords.cuda()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            # Generate the output
            out = self.main(input)
            if self.use_tanh:
                out = nn.Tanh(out)

            # Add bias to enforce locality
            if self.bias_type is not None:
                coords = self.coords.expand(out.size()[0], self.nc, 32, 32)
                out = out + coords

            # Reshape output
            out = self.reshape(out)

        return out


class DCGAN_G2(nn.Module):
    """DCGAN generator."""

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0,
                 use_tanh=False, norm=nn.BatchNorm2d):
        """Constructor."""

        super(DCGAN_G2, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main_2 = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz+3, cngf, 4, 1, 0, bias=False))
        if norm is not None:
            main.add_module('initial.{0}.batchnorm'.format(cngf),
                            norm(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.LeakyReLU())

        csize = 4
        i=0
        while csize < isize // 2:
            if i==0:
                main_2.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf+3, cngf // 2, 4, 2, 1,
                                               bias=False))
            else:
                main_2.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1,
                                               bias=False))

            if norm is not None:
                main_2.add_module('pyramid.{0}.batchnorm'.format(cngf // 2),
                                norm(cngf // 2))
            main_2.add_module('pyramid.{0}.relu'.format(cngf // 2), nn.LeakyReLU())
            cngf = cngf // 2
            csize = csize * 2
            i+=1

        # Extra layers
        for t in range(n_extra_layers):
            main_2.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=True))
            if norm is not None:
                main_2.add_module('extra-layers-{0}.{1}.batchnorm'.format(
                    t, cngf), norm(cngf))
            main_2.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.LeakyReLU())

        main_2.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, 2, 4, 2, 1, bias=True))
        if use_tanh:
            main_2.add_module('final.{0}.tanh'.format(nc), nn.Tanh())
        main_2.add_module('reshape', ReshapeSplats())
        self.main = main
        self.main_2 = main_2

    def forward(self, input, cond):
        """Forward method."""
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output1 = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            cond=cond.view(cond.size(0),cond.size(1),1,1)
            propogated = Variable(torch.ones((input.size(0), cond.size(1), input.size(2), input.size(3))), requires_grad=False).cuda()* cond
            input=torch.cat([input,propogated],1)
            output = self.main(input)
            propogated = Variable(torch.ones((output.size(0), cond.size(1), output.size(2), output.size(3))), requires_grad=False).cuda()* cond
            x=torch.cat([output,propogated],1)
            output_2 = self.main_2(x)
        return output_2

class DCGAN_G(nn.Module):
    """DCGAN generator."""

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0,
                 use_tanh=False, norm=nn.BatchNorm2d):
        """Constructor."""

        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main_2 = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz+3, cngf, 4, 1, 0, bias=False))
        if norm is not None:
            main.add_module('initial_{0}_batchnorm'.format(cngf),
                            norm(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.LeakyReLU())

        csize = 4
        i = 0
        while csize < isize // 2:
            if i == 0:
                main_2.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf+3, cngf // 2, 4, 2, 1,
                                               bias=False))
            else:
                main_2.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1,
                                               bias=False))

            if norm is not None:
                main_2.add_module('pyramid_{0}_batchnorm'.format(cngf // 2),
                                  norm(cngf // 2))
            main_2.add_module('pyramid_{0}_relu'.format(cngf // 2), nn.LeakyReLU())
            cngf = cngf // 2
            csize = csize * 2
            i+=1

        # Extra layers
        for t in range(n_extra_layers):
            main_2.add_module('extra-layers_{0}-{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=True))
            if norm is not None:
                main_2.add_module('extra-layers_{0}-{1}_batchnorm'.format(
                    t, cngf), norm(cngf))
            main_2.add_module('extra-layers_{0}-{1}_relu'.format(t, cngf),
                            nn.LeakyReLU())

        main_2.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, 1, 4, 2, 1, bias=True))
        if use_tanh:
            main_2.add_module('final_{0}_tanh'.format(nc), nn.Tanh())
        main_2.add_module('reshape', ReshapeSplats())
        self.main = main
        self.main_2 = main_2


    def forward(self, input, cond):
        """Forward method."""
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output1 = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            cond=cond.view(cond.size(0),cond.size(1),1,1)
            propogated = Variable(torch.ones((input.size(0), cond.size(1), input.size(2), input.size(3))), requires_grad=False).cuda()* cond
            input=torch.cat([input,propogated],1)
            output = self.main(input)
            propogated = Variable(torch.ones((output.size(0), cond.size(1), output.size(2), output.size(3))), requires_grad=False).cuda()* cond
            x=torch.cat([output,propogated],1)
            output_2 = self.main_2(x)
        return output_2

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.upsample = None
        if inplanes < planes:
            self.upsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = x
        if self.upsample is not None:
            residual = self.upsample(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []

        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def cosineSampleHemisphere(u1, u2):
    r = math.sqrt(u1)
    theta = 2 * math.pi * u2

    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - u1))

    return x, y, z


class _netG_resnet(nn.Module):
    def __init__(self, nz, nc, nsplats, dim=512, res_weight=0.3):
        super(_netG_resnet, self).__init__()
        self.dim = dim
        self.nc = nc  # this is the number of fields per splat (currently 6: position [x,y,z], normal [x,y,z])
        self.nsplats = nsplats
        self.nz = nz
        self.conv_dim = int(math.sqrt(nsplats))  # what goes into the resnet on each edge (x/y resolution)
        self.out_dim = int(512 * (self.conv_dim / 2) ** 2)  # what comes out of the resnet
        self.final_dim = nsplats * nc  # what will the renderer expect
        self.fc1 = nn.Linear(nz, self.conv_dim * self.conv_dim * 3)  # map noise to resnet default resolution
        self.fc2 = nn.Linear(self.out_dim, self.final_dim)  # map resnet output to splat dimensions
        self.fc3 = nn.Linear(self.final_dim, self.final_dim)  # bias addition filter

        self.resnet = ResNet(ResBasicBlock, [3, 4, 6, 3])
        print("nz {}, nc {}, nsplats {}, dim {}".format(nz, nc, nsplats, dim))
        ### nz 100, nc 6, nsplats 1024, dim 512

        self.radial_bias = np.zeros((nsplats, nc), np.float32)
        for x_i in range(self.conv_dim):
            for y_i in range(self.conv_dim):
                x, y, z = cosineSampleHemisphere(x_i, y_i)
                self.radial_bias[x_i * y_i + y_i, :3] = [x, y, z]

        self.radial_bias = Variable(torch.from_numpy(self.radial_bias.flatten()))
        if torch.cuda.is_available():
            self.radial_bias = self.radial_bias.cuda()

    def forward(self, noise):
        out = noise.view(-1, self.nz)  # remove the excess dimensions
        out = self.fc1(out)  # scale up (100) -> (224 * 224 * 3)
        out = out.view(-1, 3, self.conv_dim, self.conv_dim)  # reshape to image format (224 * 224 * 3) -> (224, 224, 3)
        # print("before res", out.size()) # torch.Size([2, 3, 32, 32])
        out = self.resnet(out)
        # print ("after res", out.size())  # torch.Size([2, 512, 16, 16])
        out = out.view(-1, self.out_dim)
        out = self.fc2(out)
        out = out.view(-1, self.nsplats, self.nc)
        # print("final size before bias:",out.size())

        filtered_bias = self.fc3(self.radial_bias).view(-1, self.nsplats, self.nc)
        # print("filtered_bias",filtered_bias.size())

        for i in range(out.size()[0]):
            out[i] = out[i] + filtered_bias
        return out


#############################################
# Discriminators
#############################################
class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf, isize, nz, use_sigmoid=0):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.use_sigmoid = use_sigmoid

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+3, ndf, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            # state size. (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 10, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            nn.Conv2d(ndf * 10, ndf * 12, 4, 1, 0, bias=True),
            nn.LeakyReLU()
            )
            # state size. (ndf*8) x 4 x 4
        self.main2 = nn.Sequential(nn.Conv2d(ndf * 12+nz, 1, 1, 1, 0, bias=True)





        )

    def forward(self, x, z, z2):
        # import ipdb; ipdb.set_trace()
        z = z.view(z.size(0), z.size(1), 1, 1)
        propogated = Variable(torch.ones((x.size(0), z.size(1),
                                          x.size(2), x.size(3))),
                              requires_grad=False).cuda() * z
        x = torch.cat([x, propogated], 1)
        x = self.main(x)
        x2 = torch.cat([x, z2], 1)
        x2= self.main2(x2)
        x2 = x2.view(-1, 1).squeeze(1)

        if self.use_sigmoid:
            return F.sigmoid(x2)
        else:
            return x2


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
class _netD_256(nn.Module):
    def __init__(self, ngpu, nc, ndf, isize, nz,use_sigmoid=0):
        super(_netD_256, self).__init__()
        self.ngpu = ngpu
        self.use_sigmoid = use_sigmoid

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+3, ndf, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf*4, ndf * 4, 4, 2, 1, bias=True),

            nn.LeakyReLU(),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 10, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 10, ndf * 12, 4, 1, 0, bias=True),

            nn.LeakyReLU(),
            nn.Conv2d(ndf * 12, 1, 1, 1, 0, bias=True)

            # nn.Conv2d(ndf * 8, ndf * 10, 4, 2, 1, bias=True),
            #
            # nn.LeakyReLU(),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 10, ndf * 10, 4, 1, 0, bias=True),
            #
            # nn.LeakyReLU(),
            # nn.Conv2d(ndf * 10, 1, 1, 1, 0, bias=True)
        )

    def forward(self, x,z):
        #import ipdb; ipdb.set_trace()
        z=z.view(z.size(0),z.size(1),1,1)
        propogated = Variable(torch.ones((x.size(0), z.size(1), x.size(2), x.size(3))), requires_grad=False).cuda()* z
        x=torch.cat([x,propogated],1)
        x = self.main(x)
        x = x.view(-1, 1).squeeze(1)

        if self.use_sigmoid:
            return F.sigmoid(x)
        else:
            return x


class _netD2(nn.Module):
    def __init__(self, ngpu, nc, ndf, isize, use_sigmoid=0):
        super(_netD2, self).__init__()
        self.ngpu = ngpu
        self.use_sigmoid = use_sigmoid

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            # state size. (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),

            nn.LeakyReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=True),

            nn.LeakyReLU(),
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1).squeeze(1)

        if self.use_sigmoid:
            return F.sigmoid(x)
        else:
            return x
class _netD_64(nn.Module):
    def __init__(self, ngpu, nc, ndf, isize, use_sigmoid=0):
        super(_netD_64, self).__init__()
        self.ngpu = ngpu
        self.use_sigmoid = use_sigmoid

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(ndf*2, ndf * 2, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # state size. (ndf*2) x 16 x 16


            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            #nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(ndf * 4, 1, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1).squeeze(1)

        if self.use_sigmoid:
            return F.sigmoid(x)
        else:
            return x


class DCGAN_D(nn.Module):
    """DCGAN Discriminator."""

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0,
                 use_sigmoid=True, norm=nn.BatchNorm2d):
        """Constructor."""
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        self.use_sigmoid = use_sigmoid
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial_conv_{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers_{0}-{1}_conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            if norm is not None:
                main.add_module('extra-layers_{0}-{1}_batchnorm'.format(
                    t, cndf), nn.BatchNorm2d(cndf))
            main.add_module('extra-layers_{0}-{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            if norm is not None:
                main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                                nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        """Forward method."""
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = output.view(-1, 1).squeeze(1)
        if self.use_sigmoid:
            return F.sigmoid(output)
        else:
            return output
