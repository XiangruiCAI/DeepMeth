
import torch.nn as nn

from models.encoder import encoder_1d, encoder_2d
from models.decoder import decoder_1d, decoder_2d


class AutoEncoder(nn.Module):

    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__()
        deconv = kwargs.get('deconv', 1)
        if deconv == 1:
            self.encoder = encoder_1d(**kwargs)
            self.decoder = decoder_1d(**kwargs)
            print("Building 1D AutoEncoder……")
        else:
            self.encoder = encoder_2d(**kwargs)
            self.decoder = decoder_2d(**kwargs)
            print("Building 2D AutoEncoder……")

    def forward(self, x):
        code = self.encoder(x)
        generate_x = self.decoder(code)
        return code, generate_x