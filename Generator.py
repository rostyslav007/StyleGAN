import torch
import torch.nn.functional as F
from torch import nn
from model import GeneratorCell, ConvNormalizedPix

class Generator(nn.Module):
    '''
    Generator architecture:
    Parameters:
          input_channels: list of generator blocks input channels
          output_channels: list of generator blocks output channels
          w_dim: dimension size for W space
          mapping: mapping layers
          img_channels: rgb image channels count
          device: device
    '''

    def __init__(self,
                 input_channels,
                 output_channels,
                 mapping,
                 w_dim,
                 img_channels=3,
                 device='cuda'):
        '''
        Creates Generator architecture
        '''
        super(Generator, self).__init__()

        self.mapping = mapping
        self.w_dim = w_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.tanh = nn.Tanh()

        # Basic noice
        self.constant_noice = nn.Parameter(torch.randn(1, w_dim, 4, 4)).to(device)

        # Gen blocks and to_rgb lists
        self.gen_blocks = nn.ModuleList()
        self.to_rgb_blocks = nn.ModuleList()

        self.start_block = GeneratorCell(input_channels[0], output_channels[0], w_dim, start_conv=False)
        self.gen_blocks.append(self.start_block)

        self.to_rgb_blocks.append(
            ConvNormalizedPix(output_channels[0], img_channels, kernel_size=1, stride=1, padding=0).to(device))

        for i in range(1, len(input_channels)):
            gen_block = GeneratorCell(input_channels[i], output_channels[i], w_dim, start_conv=True).to(device)
            self.gen_blocks.append(gen_block.to(device))

            to_rgb_block = ConvNormalizedPix(output_channels[i], img_channels, kernel_size=1, stride=1, padding=0).to(
                device).to(device)
            self.to_rgb_blocks.append(to_rgb_block)

    def fade_in(self, residual, x, alpha=None):
        '''
        Controls gradual learning process with fade-in approach
        '''

        return self.tanh((1 - alpha) * residual + alpha * x)

    def forward(self, z, steps, alpha):
        w = self.mapping(z)
        x = self.start_block(self.constant_noice, w)

        for i in range(1, steps):
            upscaled = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.gen_blocks[i](upscaled, w)

        out = self.to_rgb_blocks[steps - 1](x)
        if steps > 1:
            upscaled_rgb = self.to_rgb_blocks[steps - 1](x)
            out = self.fade_in(upscaled_rgb, out, alpha)
        else:
            out = self.tanh(out)
        return out