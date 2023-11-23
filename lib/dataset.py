import torch 
from torch.utils.data import Dataset

class FeatureSet(Dataset):

    def __init__(self, X, y, length, transform=None):
        self.X = X
        self.y = y
        self.length = length 
        self.transform = transform 

    def __getitem__(self, i):
        sample = (torch.as_tensor(self.X[i], dtype=torch.float32).nan_to_num(0), torch.as_tensor(self.y[i], dtype=torch.float32).nan_to_num(0))
        if self.transform:
            return self.transform(sample) 
        return sample 

    def __len__(self):
        return self.length 
