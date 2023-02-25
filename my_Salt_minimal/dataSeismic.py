import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import constDirectories as ct

batch_size = 256
data_mean = 0.128
data_std = 0.305

# amplitude for the noise augmentation
augm_sigma = 0.08
data_dir = ct.root  #'mnist_data'

def unnormalize(x):
    '''go from normaized data x back to the original range'''
    return x * data_std + data_mean


def normalization(x):
    return (x - data_mean) / data_std

def noise(x):
    return (x + augm_sigma * torch.randn_like(x))

class SeismicImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, label_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = label_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = self.img_labels.iloc[idx, 0] + '.png'
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


transf =  T.Compose([
            T.ToTensor(),
            T.Normalize(
                data_mean, 
                data_std, 
                inplace=False)  ])

transform = transforms.Compose([transforms.Resize((28 , 28)),
                                transforms.Grayscale(num_output_channels=1)
                                ,transforms.ToTensor()])
train_dataSeismic = SeismicImageDataset(ct.TRAIN_CSV,ct.TRAIN_IMAGE_DIR, transform)

train_loaderSeismic = DataLoader(train_dataSeismic, batch_size=64, shuffle=True)