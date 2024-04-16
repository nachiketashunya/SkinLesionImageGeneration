import torch.nn as nn

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator"""
        super(Discriminator, self).__init__()

        use_bias = True 
        kernel_size = 4
        padding = 1
        lfilters = 64
        inp_channels = 10
        norm_layer = nn.InstanceNorm2d

        self.model = nn.Sequential(
            nn.Conv2d(inp_channels, lfilters, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(lfilters * 1, lfilters * 2, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
            norm_layer(lfilters * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(lfilters * 2, lfilters * 4, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
            norm_layer(lfilters * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(lfilters * 4, lfilters * 8, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
            norm_layer(lfilters * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(lfilters * 8, lfilters * 8, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(lfilters * 16),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(lfilters * 8, 1, kernel_size=kernel_size, stride=1, padding=padding)
        )

    def forward(self, input):
        return self.model(input)

    