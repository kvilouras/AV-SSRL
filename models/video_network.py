import torch.nn as nn
from models.network_blocks import R2Plus1dBlock


class R2Plus1d(nn.Module):
    def __init__(self, depth=18, out_dim=512):
        super(R2Plus1d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        if depth == 18:
            self.conv2x = nn.Sequential(R2Plus1dBlock(64, 64), R2Plus1dBlock(64, 64))
            self.conv3x = nn.Sequential(R2Plus1dBlock(64, 128, stride=(2, 2, 2)), R2Plus1dBlock(128, 128))
            self.conv4x = nn.Sequential(R2Plus1dBlock(128, 256, stride=(2, 2, 2)), R2Plus1dBlock(256, 256))
            self.conv5x = nn.Sequential(R2Plus1dBlock(256, 512, stride=(2, 2, 2)), R2Plus1dBlock(512, 512))
        else:
            raise NotImplementedError('Only R(2+1)D-18 model is available for now.')

        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.out_dim = out_dim

    def forward(self, x, return_embs=False):
        x1 = self.conv1(x)
        x2 = self.conv2x(x1)
        x3 = self.conv3x(x2)
        x4 = self.conv4x(x3)
        x5 = self.conv5x(x4)
        x_pool = self.pool(x5)
        if return_embs:
            return dict(conv1=x1, conv2x=x2, conv3x=x3, conv4x=x4, conv5x=x5, pool=x_pool)
        else:
            return x_pool
