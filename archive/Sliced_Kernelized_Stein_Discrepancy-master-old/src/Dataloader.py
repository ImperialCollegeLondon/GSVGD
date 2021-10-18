from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchvision import datasets, transforms

class Sample_Dataset(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset
    def __len__(self):
        return self.dataset.shape[0]
    def __getitem__(self,idx):
        return self.dataset[idx,:]


class stochMNIST(datasets.MNIST):
    """ Gets a new stochastic binarization of MNIST at each call. """
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')
        img = transforms.ToTensor()(img)
        img = torch.bernoulli(img)  # stochastically binarize
        return img, target

    def get_mean_img(self):
        imgs = self.train_data.type(torch.float) / 255
        mean_img = imgs.mean(0).reshape(-1).numpy()
        return mean_img


class FixedstochMNIST(Dataset):
    # Fix the stochasticity for fair comparison
    def __init__(self,test_loader):
        for i,(data,label) in enumerate(test_loader):
            if i==0:
                self.Data=data
                self.label=label
            else:
                self.Data=torch.cat((self.Data,data),dim=0)
                self.label=torch.cat((self.label,label),dim=0)
    def __len__(self):
        return self.Data.shape[0]
    def __getitem__(self,idx):
        return self.Data[idx],self.label[idx]