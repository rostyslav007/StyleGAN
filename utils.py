import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os


def get_gradient(crit, real, fake, epsilon, step, alpha):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images, step, alpha)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty


def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    '''

    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda*gp

    return crit_loss

def get_noise(batch_size, z_dim, device):
    '''
    Function maps vectors from z dimension to w dimension:
    Parameters:
        batch_size: vector count to generate
        z_dim: dimensionof z space
        device: device name to create tensor
    '''

    noise = torch.normal(0, 1, (batch_size, z_dim), device=device)

    return noise


def show_grid(tensor, num):
    print(num)
    grid = make_grid(tensor.clamp(-1, 1).cpu().detach()*0.5 + 0.5)

    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    plt.savefig(os.path.join('results', str(num)))
