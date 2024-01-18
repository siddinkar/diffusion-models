import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

class CelebDataset(Dataset):
    def __init__(self, im_path):
        self.im_path = im_path
        self.images = []
        for i in range(10000):
            img = Image.open("img_align_celeba/{}.png".format(str(i).zfill(6)))
            im_tensor = transforms.ToTensor()(img)
            self.images.append(im_tensor)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = torch.tensor(self.images[index], dtype=torch.uint8)
        img = torch.reshape(img, (3, 178, 218))
        img = (img / 255.0).float()
        img = (2 * img) - 1

        return img

