import torch
from model import DiscriminatorCell, EqualizedConv, FinalDiscriminatorBlock
import torch.nn as nn


class Discriminator(nn.Module):
    '''
    Discriminator architecture:
    Parameters:
          input_channels: input discriminator cells channels
          output_channels: output discriminator cells chennels
          img_sizes: img sizes
          alphas: alpha coeficients
          w_dim: W space dimension
    '''
    def __init__(self,
                 input_channels,
                 output_channels,
                 w_dim,
                 device,
                 img_channels=3):
        super(Discriminator, self).__init__()

        self.w_dim = w_dim
        self.depth = len(input_channels)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.disc_blocks = nn.ModuleList()
        self.from_rgb_blocks = nn.ModuleList()

        for i in range(len(input_channels) - 1):
            disc_block = DiscriminatorCell(input_channels[i], output_channels[i], device)
            self.disc_blocks.append(disc_block)

            from_rgb_block = EqualizedConv(img_channels, input_channels[i], kernel_size=1, stride=1, padding=0)
            self.from_rgb_blocks.append(from_rgb_block.to(device))

        self.disc_blocks.append(DiscriminatorCell(input_channels[-1], output_channels[-1], device))
        self.from_rgb_blocks.append(
            EqualizedConv(img_channels, input_channels[-1], kernel_size=1, stride=1, padding=0).to(device))

        #self.final_block = FinalDiscriminatorBlock(input_channels[-1] + 1, output_channels[-1], w_dim)
        self.final_block = FinalDiscriminatorBlock(input_channels[-1], output_channels[-1], w_dim)

    def fade_in(self, downscaled, x, alpha):
        '''
        Fades in images
        '''
        return (1 - alpha) * downscaled + alpha * x

    def minibatch_statistics(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, steps, alpha):
        pos = self.depth - steps

        x_copy = x.clone()
        x = self.from_rgb_blocks[pos](x)
        if steps > 1:
            down_block = self.avg_pool(x_copy)
            down_from_rgb = self.from_rgb_blocks[pos + 1](down_block)
            x = self.disc_blocks[pos](x)
            down = self.avg_pool(x)
            x = self.fade_in(down_from_rgb, down, alpha)

        for i in range(pos + 1, self.depth - 1):
            x = self.disc_blocks[i](x)
            x = self.avg_pool(x)

        x = self.disc_blocks[self.depth - 1](x)

        #x = self.minibatch_statistics(x)
        x = self.final_block(x)

        return x
