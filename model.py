import torch
from torch import nn
import torch.nn.functional as F


class EqualizedConv(nn.Module):
    '''
    Equalized Linear layer:
    Parameters:
        in: input channels count
        out: output channels count
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(EqualizedConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize weights
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    '''
    makes tensor channels magnitude equal 1 for every pixel channel
    '''
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        b, c, h, w = x.shape
        channel_sum = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) / c + self.epsilon)
        x = x / channel_sum

        return x


class ConvNormalizedPix(nn.Module):
    '''
    Convolution with pixel norm
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(ConvNormalizedPix, self).__init__()
        self.equalized_conv = EqualizedConv(in_channels, out_channels, kernel_size, stride, padding, gain)
        self.pix_norm = PixelNorm()

    def forward(self, x):
        return self.pix_norm(self.equalized_conv(x))


class EqualizedLinear(nn.Module):
    '''
    Equalized Linear layer
    Parameters:
            in_dim: input dimension
            out_dim: output dimension
    '''

    def __init__(self, in_dim, out_dim, gain=2):
        super(EqualizedLinear, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.bias = self.linear.bias
        self.linear.bias = None

        self.scale = (gain / in_dim) ** 0.5

        # initialize linear layer
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class Mapping(nn.Module):
    '''
    class for mapping part:
      Parameters:
            z_dim: dimention of latent space vector
            hidden_dim: dimention of hidden space
    '''

    def __init__(self, z_dim, hidden_dim):
        super(Mapping, self).__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.w_dim = z_dim

        inner_layers = nn.ModuleList()
        inner_layers.append(EqualizedLinear(z_dim, hidden_dim))
        for i in range(4):
            inner_layers.append(EqualizedLinear(hidden_dim, hidden_dim))
            inner_layers.append(nn.LeakyReLU(0.1))

        inner_layers.append(EqualizedLinear(hidden_dim, self.w_dim))
        inner_layers.append(nn.ReLU())

        self.mapping = nn.Sequential(*inner_layers)

    def forward(self, z):
        '''
        Returns mapped vectors from z to w space
        Prameters:
              z: vectors of latent z space
        '''
        return self.mapping(z)


class NoiceChannel(nn.Module):
    '''
    Noice with weights channel estimate
    '''

    def __init__(self, in_channels):
        super(NoiceChannel, self).__init__()

        self.weights = nn.Parameter(torch.normal(0, 1, size=(1, in_channels, 1, 1))).cuda()*0.01

    def forward(self, images):

        b, c, h, w = images.shape

        noice_shape = (b, 1, h, w)

        noice = torch.randn(size=noice_shape).cuda()

        return images + self.weights * noice


class AdaIN(nn.Module):
    '''
    Adaptive instance normalization layer
    Parameters:
        in_channels: input tensor channels count
        w_dim: dimension of W space vector
    '''

    def __init__(self, w_dim, in_channels):
        super(AdaIN, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(in_channels)
        self.style_scale_transform = EqualizedLinear(w_dim, in_channels)
        self.style_shift_transform = EqualizedLinear(w_dim, in_channels)

    def forward(self, x, w):
        '''
        Performs AdaIN normalization
        '''

        normalized_image = self.instance_norm(x)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]

        transformed_image = style_scale * normalized_image + style_shift

        return transformed_image


class GeneratorCell(nn.Module):
      '''
      Generator progressive block
      Parameters:
            input_channels: Conv2d input channels count
            out_channels: Conv2d output channels count
            parameters from vector from W space
            w_dim: dimension of W space
      '''
      def __init__(self, input_channels, out_channels, w_dim, start_conv=False):
          '''
          Saves object parameters for progressive block
          '''
          super(GeneratorCell, self).__init__()

          self.start_conv = start_conv

          self.add_noice1 = NoiceChannel(input_channels)
          self.add_noice2 = NoiceChannel(out_channels)

          self.conv0 = ConvNormalizedPix(input_channels, input_channels)
          self.conv1 = ConvNormalizedPix(input_channels, out_channels)
          self.leaky_relu = nn.LeakyReLU(0.2)

          self.adain1 = AdaIN(w_dim, input_channels)
          self.adain2 = AdaIN(w_dim, out_channels)

      def forward(self, x, w):
          '''
          Forward path for progressive block
          '''
          if self.start_conv:
              x = self.leaky_relu(self.conv0(x))
          x = self.add_noice1(x)

          x = self.adain1(x, w)
          x = self.leaky_relu(self.conv1(x))

          x = self.add_noice2(x)
          x = self.adain2(x, w)

          return x


class DiscriminatorCell(nn.Module):
    '''
    Discriminator block:
    Parameters:
          in_channels: input channels count
          out_channels: output cahnnels count
          alpha: fade in parameter
          down: perform downscale or not
    '''

    def __init__(self, in_channels, out_channels, device):
        super(DiscriminatorCell, self).__init__()

        self.conv1 = EqualizedConv(in_channels, out_channels)
        self.conv2 = EqualizedConv(out_channels, out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


class FinalDiscriminatorBlock(nn.Module):
    '''
    Discriminator last predictive layers
    '''

    def __init__(self, input, output, w_dim, kernel_size=4, stride=1, padding=0):
        super(FinalDiscriminatorBlock, self).__init__()
        self.conv1 = EqualizedConv(input, output, kernel_size=3, stride=1, padding=1)
        self.conv2 = EqualizedConv(output, output, kernel_size, stride, padding)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU(0.01)
        self.linear = EqualizedLinear(w_dim, 1)

    def forward(self, x):
        return self.linear(self.flatten(self.relu(self.conv2(self.relu(self.conv1(x))))))




