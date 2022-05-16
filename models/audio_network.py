import torch.nn as nn
from models.network_blocks import Conv2dBlock


__all__ = [
    'Conv2DNet'
]


class Conv2DNet(nn.Module):
    def __init__(self, out_dim=512, depth=10):
        super(Conv2DNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block1 = Conv2dBlock(64, 64, stride=(2, 2))
        self.block2 = Conv2dBlock(64, 128, stride=(2, 2))
        self.block3 = Conv2dBlock(128, 256, stride=(2, 2))
        self.block4 = Conv2dBlock(256, 512)

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.out_dim = out_dim

    def forward(self, x, return_embs=False):
        x1 = self.conv1(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x_pool = self.pool(x5)
        if return_embs:
            return dict(conv1=x1, block1=x2, block2=x3, block3=x4, block4=x5, pool=x_pool)
        else:
            return x_pool

