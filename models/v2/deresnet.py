from typing import Type, Union

from torch import nn


def deconv3x(deconv: Union[Type[nn.ConvTranspose1d], Type[nn.ConvTranspose2d]], in_planes, out_planes, stride=1,
             groups=1,
             dilation=1):
    """3x3 convolution with padding"""
    return deconv(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=dilation, groups=groups, bias=False, dilation=dilation)


def deconv1x(deconv: Union[Type[nn.ConvTranspose1d], Type[nn.ConvTranspose2d]], in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return deconv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DeBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, conv, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d):
        super(DeBasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.deconv1 = deconv3x(conv, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = deconv3x(conv, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeResNet(nn.Module):

    def __init__(self):
        super().__init__()
