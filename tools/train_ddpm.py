import torch
import yaml
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
from torch.optim import Adam
from data.cifar_dataset import CifarDataset
from torch.utils.data import DataLoader
from models.unet import Unet
from schedulers.linear_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # read config file
    with open("../" + args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    scheduler = LinearNoiseScheduler(timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    training_batch = CifarDataset(dataset_config['im_path'])
    loader = DataLoader(training_batch, batch_size=train_config['batch_size'])

    model = Unet(model_config).to(device)
    model.train()

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'], train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))

    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.KLDivLoss()

    for epoch in range(num_epochs):
        losses = []
        for img in tqdm(loader):
            optimizer.zero_grad()
            img = (img[0] / 256.0).float().to(device)

            # random noise
            noise = torch.randn_like(img).to(device)

            # random timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (img.shape[0],)).to(device)

            # add noise
            noisy_img = scheduler.add_noise(img, noise, t)
            noise_pred = model(noisy_img, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss: {:.4f}'.format(
            epoch + 1,
            np.mean(losses)
        ))
        torch.save(model.state_dict(), os.path.join("../",
                                                    train_config['task_name'],
                                                    train_config['ckpt_name']))

    print('Done Training...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)