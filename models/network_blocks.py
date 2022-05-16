import torch.nn as nn


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(Conv2dBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class R2Plus1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super(R2Plus1dBlock, self).__init__()
        spatial_stride = (1, stride[1], stride[2])
        temporal_stride = (stride[0], 1, 1)

        self.spatial_conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                                       stride=spatial_stride, padding=(0, 1, 1), bias=False)
        self.spatial_bn1 = nn.BatchNorm3d(out_channels)
        self.temporal_conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1),
                                        stride=temporal_stride, padding=(1, 0, 0), bias=False)
        self.temporal_bn1 = nn.BatchNorm3d(out_channels)

        self.spatial_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3),
                                       stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.spatial_bn2 = nn.BatchNorm3d(out_channels)
        self.temporal_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1),
                                        stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.out_bn = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # residual path
        if in_channels != out_channels or any([s != 1 for s in stride]):
            self.res = True
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1),
                                      stride=stride, padding=(0, 0, 0), bias=False)
        else:
            self.res = False

    def forward(self, x):
        x_main = self.relu(self.spatial_bn1(self.spatial_conv1(x)))
        x_main = self.relu(self.temporal_bn1(self.temporal_conv1(x_main)))
        x_main = self.temporal_conv2(self.relu(self.spatial_bn2(self.spatial_conv2(x_main))))

        x_res = self.res_conv(x) if self.res else x
        x_out = self.relu(self.out_bn(x_main + x_res))

        return x_out

