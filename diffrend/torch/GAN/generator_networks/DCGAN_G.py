import torch
from torch import nn
from torch.autograd import Variable
from diffrend.torch.GAN.generator_networks.common import ReshapeSplats

class DCGAN_G(nn.Module):
    """DCGAN generator."""

    def __init__(self, isize, nz, nc, ngf, ngpu, cond_size=3, n_extra_layers=0,
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
                        nn.ConvTranspose2d(nz+cond_size, cngf, 4, 1, 0, bias=False))
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
                            nn.ConvTranspose2d(cngf+cond_size, cngf // 2, 4, 2, 1,
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
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=True))
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