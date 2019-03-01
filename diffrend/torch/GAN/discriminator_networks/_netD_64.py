import torch
from torch import nn
import torch.nn.functional as F

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