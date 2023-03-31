import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
import os
import directories as dl

#Seismic image parameters
data_mean = 0.4746
data_std = 0.1526

# amplitude for the noise augmentation
augm_sigma = 0.08

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

# Custom Dataset Partition
batch = 64
resiz=(28,28)
workers = 8

transform = T.Compose([T.Resize(resiz),
                       T.Grayscale(num_output_channels=1),
                       T.RandomHorizontalFlip(),
                       T.ToTensor()
                       #,T.Normalize(data_mean, data_std, inplace=False) 
                     ])

transformTest = T.Compose([T.Resize(resiz),
                       T.Grayscale(num_output_channels=1),
                       T.ToTensor()
                       #,T.Normalize(data_mean, data_std, inplace=False) 
                     ])

train_dataset = SeismicImageDataset(dl.TRAIN_NOT_NULLCSV,dl.TRAIN_IMAGE_DIR, transform)
test_dataset  = SeismicImageDataset(dl.TEST_NOT_NULLCSV, dl.TRAIN_IMAGE_DIR, transformTest)

# create batches loader
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False , num_workers=workers)
test_loader =  DataLoader(test_dataset , batch_size=batch, shuffle=False, num_workers=workers)

split = len(test_dataset)
# Sample a fixed batch of validation examples (split=validationSamples)
val_x, val_l = zip(*list(test_dataset[i] for i in range(split)))
val_x = torch.stack(val_x, 0).cuda()
val_l = torch.LongTensor(val_l).cuda()