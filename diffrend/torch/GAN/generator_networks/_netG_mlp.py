import torch
from torch import nn

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