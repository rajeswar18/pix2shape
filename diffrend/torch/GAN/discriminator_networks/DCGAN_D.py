import torch
from torch import nn
import torch.nn.functional as F


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