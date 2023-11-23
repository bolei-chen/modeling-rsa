import torch
import torch.nn as nn
import torch.optim as optim 

from tqdm import tqdm 
from matplotlib import pyplot as plt 


class FFNN(nn.Module):

    def __init__(self, device):
        super(FFNN, self).__init__()
        self.log = {
            'training loss' : [],
            'validation loss' : [],
        } 
        self.device = device 
        self.input = nn.Linear(8, 36) 
        torch.nn.init.xavier_uniform(self.input.weight)
        self.l1 = nn.Linear(36, 36) 
        self.l2 = nn.Linear(36, 36) 
        self.l3 = nn.Linear(36, 36) 
        self.l4 = nn.Linear(36, 36) 
        self.l5 = nn.Linear(36, 36) 
        self.l6 = nn.Linear(36, 36) 
        self.l7 = nn.Linear(36, 36) 
        self.output = nn.Linear(36, 3) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x.float())
        x = self.relu(self.l1(x)) 
        x = self.relu(self.l2(x)) 
        x = self.relu(self.l3(x)) 
        x = self.relu(self.l4(x)) 
        x = self.relu(self.l5(x)) 
        x = self.relu(self.l6(x)) 
        x = self.relu(self.l7(x)) 
        x = self.output(x)
        return x

    def fit(self, epochs, lr, loader_train, loader_val):
         
        optimizer = optim.Adam(self.parameters(), lr=lr)
         
        for i in tqdm(range(epochs)):
            for xs, ys in loader_train: 
                xs = xs.to(self.device) 
                ys = ys.to(self.device) 
                optimizer.zero_grad()   
                y_pred = self(xs)
                criterion = torch.nn.MSELoss() 
                loss = criterion(y_pred, ys)
                loss.backward()
                optimizer.step()    
                self.log['training loss'].append(loss.item()) 
         
            self.eval() 

            with torch.no_grad():
                for xs, ys in loader_val:
                    xs = xs.to(self.device)
                    ys = ys.to(self.device) 
                     
                    y_hats = self(xs)
                    criterion = torch.nn.MSELoss() 
                    loss = criterion(y_hats, ys)
                    self.log['validation loss'].append(loss.item()) 
            self.train() 

    def evaluate(self):
        plt.figure(figsize=(15, 10)) 
        plt.subplot(2, 1, 1)
        plt.title('training loss') 
        plt.plot(self.log['training loss']) 
        plt.xlabel('batch') 
        plt.ylabel('loss') 
         
        plt.subplot(2, 1, 2)
        plt.title('validation loss') 
        plt.plot(self.log['validation loss']) 
        plt.xlabel('batch') 
        plt.ylabel('loss') 
                          