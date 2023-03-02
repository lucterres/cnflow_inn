import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os
import directories as ct

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
        label = 1            #self.img_labels.iloc[idx, 1]
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


validationSamples = 256
# Sample a fixed batch of 1024 validation examples
val_x, val_l = zip(*list(train_dataSeismic[i] for i in range(validationSamples)))
val_x = torch.stack(val_x, 0).cuda()
val_l = torch.LongTensor(val_l).cuda()

# Custom Dataset Partition
dataset = train_dataSeismic
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

split=256

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader      = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

# Exclude the validation batch from the training data
#train_dataSeismic.data = train_dataSeismic.data[validationSamples:]
#train_dataSeismic.targets = train_dataSeismic.targets[validationSamples:]
# Add the noise-augmentation to the (non-validation) training data:
#train_dataSeismic.transform = T.Compose([train_dataSeismic.transform, T.Lambda(noise)])    

train_loaderSeismic = DataLoader(train_dataSeismic, batch_size=64, shuffle=True)