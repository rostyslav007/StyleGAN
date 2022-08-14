from model import Mapping
from Generator import Generator
from Discriminator import Discriminator
from Dataset_loader import CelebaDataset
from torch.utils.data import DataLoader
from utils import *
from config import *
import torch
from torch import optim
from torchvision import transforms

torch.autograd.set_detect_anomaly(True)

mapping = Mapping(z_dim, w_dim).to(device)

steps = len(alphas)
gen = Generator(input_channels, output_channels, mapping, w_dim, device=device).to(device)
gen_opt = optim.Adam(gen.parameters(), lr=a, betas=(beta1, beta2))

disc = Discriminator(output_channels[::-1], input_channels[::-1], w_dim, device=device).to(device)
disc_opt = optim.Adam(disc.parameters(), lr=a, betas=(beta1, beta2))

n_critic = 5

disc_hist = []
gen_hist = []

for step in range(1, steps):
    batch_size = batch_sizes[step - 1]
    img_size = img_sizes[step - 1]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])
    dataset = CelebaDataset(dir_name, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    part = seen_images // 8
    count = 0
    while count < 2 * seen_images:
        alpha = alphas[min(7, int(count // part))]

        mean_iteration_critic_loss = 0
        for _ in range(n_critic):
            real = next(iter(data_loader)).to(device)
            batch_size = real.shape[0]
            count += batch_size
            # Update critic
            fake_noise = get_noise(batch_size, z_dim, device=device)

            fake = gen(fake_noise, step, alpha)
            crit_fake_pred = disc(fake.detach(), step, alpha)
            crit_real_pred = disc(real, step, alpha)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(disc, real, fake.detach(), epsilon, step, alpha)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
            # Update gradients
            disc_opt.zero_grad()

            crit_loss.backward(retain_graph=True)
            # Update optimizer
            disc_opt.step()

            mean_iteration_critic_loss += crit_loss.item() / n_critic

        fake_noise = get_noise(batch_size, z_dim, device=device)
        fake = gen(fake_noise, step, alpha)
        crit_fake_pred = disc(fake.detach(), step, alpha)
        gen_loss = get_gen_loss(crit_fake_pred)
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        disc_hist.append(mean_iteration_critic_loss)
        gen_hist.append(gen_loss.item())

        print('disc_loss : ', disc_hist[-1], 'gen_loss : ', gen_hist[-1])

        if len(disc_hist) % 100 == 1:
            print(len(disc_hist))
            n = get_noise(batch_size, z_dim, device)
            fakes = gen(n, step, alpha) * 0.5 + 0.5
            show_grid(fakes, len(disc_hist))

