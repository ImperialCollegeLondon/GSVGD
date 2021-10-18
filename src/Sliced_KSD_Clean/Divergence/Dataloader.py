from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch
import pandas as pd
from torchvision import datasets, transforms
from PIL import Image

def UCI_preprocess(path,split_ratio=[0.9,0.1],flag_normalize=True):

    Data = pd.read_excel(path)
    Data = Data.as_matrix()

    X_input = Data[:, range(Data.shape[1] - 1)]
    Y_input = Data[:, Data.shape[1] - 1]

    train_ratio = split_ratio[0]
    permutation = np.arange(X_input.shape[0])
    random.shuffle(permutation)

    size_train = int(np.round(X_input.shape[0] * train_ratio))
    idx_train = permutation[0:size_train]
    idx_test = permutation[size_train:]
    # Split the data
    X_train, Y_train = X_input[idx_train, :], Y_input[idx_train]
    X_test, Y_test = X_input[idx_test, :], Y_input[idx_test]
    if flag_normalize: # normalize the data to 0 mean 1 variance
        mean_X_train, mean_Y_train = np.mean(X_train, axis=0), np.mean(Y_train)
        std_X_train, std_Y_train = np.std(X_train, axis=0), np.std(Y_train)

        std_X_train=std_X_train+1e-8
        std_Y_train=std_Y_train+1e-8

        #mean_X_test, mean_Y_test = None,None,#np.mean(X_test, axis=0), np.mean(Y_test)
        #std_X_test, std_Y_test = None,None#np.std(X_test, axis=0), np.std(Y_test)

        # std_X_test = std_X_test + 1e-8
        # std_Y_test = std_Y_test + 1e-8


        X_train = (X_train - np.full(X_train.shape, mean_X_train)) / np.full(X_train.shape, std_X_train)
        Y_train = (Y_train - mean_Y_train) / std_Y_train

        X_test = (X_test - np.full(X_test.shape, mean_X_train)) / np.full(X_test.shape, std_X_train)

    state_dict={}
    state_dict['mean_X_train'],state_dict['mean_Y_train']=mean_X_train,mean_Y_train
    state_dict['std_X_train'],state_dict['std_Y_train']=std_X_train,std_Y_train
    #state_dict['mean_X_test'],state_dict['mean_Y_test']=mean_X_test,mean_Y_test
    #state_dict['std_X_test'],state_dict['std_Y_test']=std_X_test,std_Y_test
    state_dict['X_train'],state_dict['Y_train'],state_dict['X_test'],state_dict['Y_test']= \
        X_train,Y_train,X_test,Y_test
    return state_dict

class UCI_Dataset(Dataset):

    def __init__(self,state_dict,flag_test=False):
        #raise NotImplementedError


        if flag_test:
            self.X_data=torch.from_numpy(state_dict['X_test']).float().cuda()
            self.Y_data=torch.from_numpy(state_dict['Y_test']).float().cuda()
            self.Y_cp=state_dict['Y_test']
        else:
            self.X_data=torch.from_numpy(state_dict['X_train']).float().cuda()
            self.Y_data=torch.from_numpy(state_dict['Y_train']).float().cuda()
            self.Y_cp=state_dict['Y_train']

    def __len__(self):
        return len(self.Y_cp)
    def __getitem__(self, idx):
        return self.X_data[idx,:],self.Y_data[idx]