
import os
import torch
import cv2
import numpy as np


class Dataset:
    def __init__(self, path, files):
        self.path = path
        self.files = files
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pth = self.files[idx]
        for_label = pth.split("_")[0]
        pth = self.path + r"\\" + str(for_label) + "\\" + str(pth)

        if for_label == "airplane":
            label = 0
        elif for_label == "car":
            label = 1
        elif for_label == "cat":
            label = 2
        elif for_label == "dog":
            label = 3
        elif for_label == "flower":
            label = 4
        elif for_label == "fruit":
            label = 5
        elif for_label == "motorbike":
            label = 6
        else:
            label = 7
        
        im = cv2.imread(pth)
        im = cv2.resize(im, (224, 224))
        return torch.tensor(im, dtype = torch.float32), torch.tensor(label, dtype = torch.float32).type(torch.LongTensor)

if __name__ == "__main__":
    path = r"Y:\\pytorch\\natural_images"
    files = [] 
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(filename))
    try_data = Dataset(path, files)

    im, lab = (try_data[1000])
    #print(im.shape)
    print(type(im))