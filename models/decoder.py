import torch.nn as nn
from models.deresnet.deresnet_1d import deconvresnet18_1d
from models.deresnet.deresnet_2d import deconvresnet18_2d


class Decoder1D(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder1D, self).__init__()
        encode_dim = kwargs.get('encode_dim', 50)
        self.reads_height = kwargs.get('region_height', 1000)
        self.decodeFC = nn.Sequential(
            nn.Linear(encode_dim, 256 * self.reads_height))
        self.deresnet = deconvresnet18_1d()

    def forward(self, code):
        x = self.decodeFC(code)
        x = x.view(x.size(0), 256, self.reads_height, 1)
        x = self.deresnet(x)
        return x


class Decoder2D(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder2D, self).__init__()
        encode_dim = kwargs.get('encode_dim', 50)
        self.decodeFC = nn.Sequential(
            nn.Linear(encode_dim, 512 * 32 * 2))
        self.deresnet = deconvresnet18_2d()

    def forward(self, code):
        x = self.decodeFC(code)
        x = x.view(x.size(0), 512, 32, 2)
        x = self.deresnet(x)
        return x


def decoder_1d(**kwargs):
    model = Decoder1D(**kwargs)
    return model


def decoder_2d(**kwargs):
    model = Decoder2D(**kwargs)
    return model

