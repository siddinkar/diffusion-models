import torch
from torch.utils.data.dataset import Dataset
import torchvision
import cv2
import numpy as np

def unpickle(file):
    import pickle
    with open("/home/sid/diffusion-models/data/cifar-10-batches-py/data_batch_5", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CifarDataset(Dataset):
    def __init__(self, im_path):
        self.im_path = im_path
        self.dataset = unpickle(self.im_path)
        self.images = self.dataset[b'data']
        self.labels = self.dataset[b'labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = torch.tensor(self.images[index], dtype=torch.uint8)
        img = torch.reshape(img, (3, 32, 32))
        img = (img / 255.0).float()
        img = (2 * img) - 1

        return img

