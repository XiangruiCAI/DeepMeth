import torch.nn as nn


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, output_padding=output_padding, bias=False)


class DeconvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, output_padding=0, upsample=None):
        super(DeconvBasicBlock, self).__init__()
        self.conv1 = deconv3x3(inplanes, planes, stride, output_padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = deconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)
        return out


class DeconvBottleneck(nn.Module):
    def __init__(self, inplanes, planes, expansion=2, stride=1, upsample=None):
        self.expansion = expansion
        super(DeconvBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if stride==1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3,
                                            stride=stride, bias=False, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            residual = self.upsample
        out += residual
        out = self.relu(out)
        return out


class DeconvResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 512
        super(DeconvResNet, self).__init__()
        self.layer3 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[1], stride=2, output_padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0])

        self.conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        self.tanh = nn.Tanh()

    def _make_layer(self, block, planes, blocks, stride=1, output_padding=0):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=False, output_padding=output_padding),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, output_padding, upsample))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.layer0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.tanh(x)
        return x


def deconvresnet18_2d():
    model = DeconvResNet(DeconvBasicBlock, [2, 2, 2, 2])
    return model





