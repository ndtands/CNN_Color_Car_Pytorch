from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import write_json       
from configs import *

def Color_Dataloader(dir_dataset: str, batch_size: int) -> tuple:
    """
    > The function takes in the directory of the dataset and the batch size, and returns a tuple of
    train and test dataloaders
    
    :param dir_dataset: The directory where the dataset is stored
    :type dir_dataset: str
    :param batch_size: The number of images to be passed through the network at once
    :type batch_size: int
    :return: A tuple of train_dataloader and test_dataloader
    """
    dir_train = f'{dir_dataset}/test'
    dir_test = f'{dir_dataset}/train'
    data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor() 
    ])
    train_data = datasets.ImageFolder(root=dir_train, 
                                      transform=data_transform,
                                      target_transform=None) 
    test_data = datasets.ImageFolder(root=dir_test, 
                                     transform=transforms.Compose([
                                                transforms.Resize(size=(224, 224)),
                                                transforms.ToTensor() 
                                            ])
                                                                            )
    train_dataloader = DataLoader(dataset=train_data, 
                                  batch_size=batch_size, 
                                  shuffle=True) 
    test_dataloader = DataLoader(dataset=test_data, 
                                 batch_size=batch_size, 
                                 shuffle=False) 
    class_dict = train_data.class_to_idx
    IDX2TAG = {v:k for k,v in class_dict.items()}
    write_json(IDX2TAG_NAME, IDX2TAG)
    return  (train_dataloader, test_dataloader)
