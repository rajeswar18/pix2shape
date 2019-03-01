import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from diffrend.torch.GAN.generator_networks.common import ReshapeSplats


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