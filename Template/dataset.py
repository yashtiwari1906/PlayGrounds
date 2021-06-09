import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


class PrepareDataset:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
    def return_data_loaders(self):
        train = pd.read_csv(self.path,dtype = np.float32)

        # split data into features(pixels) and labels(numbers from 0 to 9)
        targets_numpy = train.label.values
        features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

        # train test split. Size of train data is 80% and size of test data is 20%. 
        features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                                    targets_numpy,
                                                                                    test_size = 0.2,
                                                                                    random_state = 42) 

        # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
        featuresTrain = torch.from_numpy(features_train)
        targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

        # create feature and targets tensor for test set.
        featuresTest = torch.from_numpy(features_test)
        targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

        # batch_size, epoch and iteration
       

        # Pytorch train and test sets
        train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
        test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size = self.batch_size, shuffle = False)
        test_loader = torch.utils.data.DataLoader(test, batch_size = self.batch_size, shuffle = False)
        return train_loader, test_loader

