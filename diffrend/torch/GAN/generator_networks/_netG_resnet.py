import torch
from torch import nn
from torch.autograd import Variable
from diffrend.torch.GAN.generator_networks.common import ReshapeSplats
import math
import numpy as np


def cosineSampleHemisphere(u1, u2):
    r = math.sqrt(u1)
    theta = 2 * math.pi * u2

    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - u1))

    return x, y, z

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