import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf, isize, nz, cond_size=3, use_sigmoid=0):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.use_sigmoid = use_sigmoid

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+cond_size, ndf, 4, 2, 1, bias=True),
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