import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class FlowerDataset:
    def __init__(self, path, files, labels, task = "TRAIN"):
        self.path = path
        self.task = task 
        self.files, self.labels = files, labels
    
    
    def __len__(self):
        return len(self.files)

    
    def __getitem__(self, idx):
        pth = self.path + "//" + self.files[idx]
        label = int(self.labels[idx])
        im = cv2.imread(pth)
        #im = self.transform(im, 224, 224)
        im = cv2.resize(im, (224, 224))
        label = torch.tensor(label, dtype = torch.float32)
        return im, label

    def transform(self,im, x, y):

        if self.task == "TRAIN":
            data_transpose = transforms.Compose([
                transforms.Resize(size = (x, y)),
                transforms.RandomRotation(degrees = (-20, +20)),
                transforms.ToTensor()
                #transforms.Normalize([, , ], [, , ])
            ])
        else:
            data_transpose = transforms.Compose([
                transforms.Resize(size = (x, y)),
                transforms.ToTensor()
                #transforms.Normalize([, , ], [, , ])
            ])

        return data_transpose(im)




if __name__ == "__main__":

    path = r"Y:\\pytorch\\flower classification\\flower_data\\flower_data\\train"
    dataset = FlowerDataset(path, task = "TRAIN")
    im, label = dataset[0]
    print("shape of images : {}".format(im.shape))
    print("type of images : {}".format(type(im)))
    print("label type: {}".format(type(label)))
