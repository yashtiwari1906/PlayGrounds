import torch.nn as nn
import torch
from torchvision import models
from collections import OrderedDict
from torchvision import models
#from torchsummary import summary

class ResnetModel(nn.Module):
    def __init__(self):
        super(ResnetModel, self).__init__()
        self.base_model = models.resnet18(pretrained = True)
        n_inputs = self.base_model.fc.in_features
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(n_inputs, 512))]))
        self.base_model.fc = classifier
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        out = self.base_model(x)
        out = self.fc2(out)
        return out 

if __name__ == "__main__":
   
    model = ResnetModel()
    print(model.to(torch.device("cpu")))







