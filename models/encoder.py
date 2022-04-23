import torch.nn as nn
from models.resnet.resnet_1d import resnet18_1d
from models.resnet.resnet_2d import resnet18_2d


class Encoder1D(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder1D, self).__init__()
        encode_dim = kwargs.get('encode_dim', 50)
        self.resnet = resnet18_1d()
        self.region_height = 1000
        if 'region_height' in kwargs:
            self.region_height = kwargs['region_height']
        self.encodeFC = nn.Sequential(
            nn.Linear(256 * self.region_height, encode_dim))

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        code = self.encodeFC(x)
        return code


class Encoder2D(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder2D, self).__init__()
        encode_dim = kwargs.get('encode_dim', 50)
        self.resnet = resnet18_2d()
        self.encodeFC = nn.Sequential(
            nn.Linear(512 * 32 * 2, encode_dim))

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        code = self.encodeFC(x)
        return code


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        conv = kwargs.get('deconv', 1)
        if conv == 1:
            self.encoder = encoder_1d(**kwargs)
        else:
            self.encoder = encoder_2d(**kwargs)

    def forward(self, x):
        code = self.encoder(x)
        return code


def encoder_1d(**kwargs):
    model = Encoder1D(**kwargs)
    return model


def encoder_2d(**kwargs):
    model = Encoder2D(**kwargs)
    return model
