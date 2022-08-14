import numpy as np
import torch

dir_name = 'data/archive/img_align_celeba/img_align_celeba/'
img_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
batch_sizes = [128, 128, 64, 64, 32, 16, 8, 4]
input_channels = [256, 256, 256, 256, 256, 128, 64, 32]
output_channels = [256, 256, 256, 256, 128, 64, 32, 16]
alphas = list(np.linspace(0.1, 1, num=len(img_sizes), endpoint=True))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
a = 0.0001
c_lambda = 10
beta1 = 0
beta2 = 0.99
seen_images = 4*1e5
z_dim = 256
w_dim = z_dim