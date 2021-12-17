
from torch import nn, Tensor

from models.v2.resnet import resnet18_1d


class Encoder(nn.Module):

    def __init__(self, conv: nn.Module, transformer: nn.Module):
        super().__init__()
        self.conv = conv
        self.transformer = transformer

    def forward(self, x: Tensor):
        features = self.conv(x)
        features = Tensor.view(features, features.size(0), -1)
        code = self.transformer(features)
        return code

    @classmethod
    def from_config(cls, config) -> 'Encoder':
        pass


class ResNetEncoder(Encoder):

    def __init__(self, transformer: nn.Module):
        conv = resnet18_1d()

        super().__init__(conv, transformer)
