import torch
import torchvision
import argparse
import yaml
import os
import cv2
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet import Unet
from schedulers.linear_scheduler import LinearNoiseScheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, model_config, diffusion_config):
    xt = torch.normal(0.0, 1., (
        train_config['num_samples'],
        model_config['im_channels'],
        model_config['im_size'],
        model_config['im_size']
    )).to(device)
    xt = torch.clamp(xt, -1., 1.)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        im = torch.clamp(xt, -1., 1.).detach().cpu()
        im = (im + 1) / 2
        grid = make_grid(im, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        cv2.imshow("s", cv2.hconcat(
            [
                np.array(im[0].T.detach().cpu()),
                np.array(((x0_pred + 1) / 2)[0].T.detach().cpu())
            ]
        ))
        cv2.waitKey(0)

        if not os.path.exists(os.path.join("../", train_config['task_name'], 'samples')):
            os.mkdir(os.path.join("../", train_config['task_name'], 'samples'))
        img.save(os.path.join("../", train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
    with open("../" + args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']

    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join("../",
                                                  train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))

    model.eval()
    scheduler = LinearNoiseScheduler(timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
