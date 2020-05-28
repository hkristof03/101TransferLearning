from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader, sampler

import os
# Image manipulations
from PIL import Image


def get_data_loaders():
    """
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Location of data
    datadir = base_path + '/datasets/'
    traindir = datadir + 'train/'
    validdir = datadir + 'valid/'
    testdir = datadir + 'test/'
    # Change to fit hardware
    batch_size = 128
    # Image transformations
    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224), # Imagenet standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]) # Imagenet standards
        ]),
        # Validation does not use augmentation
        'val':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Datasets from each folder
    data = {
        'train':
        datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'val':
        datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
        'test':
        datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }
    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }

    return dataloaders
