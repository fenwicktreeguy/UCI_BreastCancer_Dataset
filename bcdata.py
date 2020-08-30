import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os
from torch import dtype


#HYPERPARAMETERS FOR NETWORK
batch_sz = 10
num_wrk = 5
num_epoch = 70
learning_rate = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BCTrainDataset(torch.utils.data.Dataset):
    def __init__(self, transform = None):
        f = os.path.join('./data/MNIST/bcwrapper/breast-cancer-wisconsin.data')
        ln1 = []
        self.samples = []
        self.labels = []
        self.ID = []
        with open(f, 'r') as opn:
            for l in opn:
                ln1.extend(l.strip().split(','))
            for c in range(len(ln1)): 
                if ln1[c] == '?': 
                    ln1[c] = '0'
            
            ln1 = list(map(int, ln1))
            sz= int(len(ln1)/2)
            print(len(ln1))
            for i in range(0, sz, 11):
                tup_v = tuple(ln1[i+1:i+10])
                self.samples.append(tup_v)
                if(ln1[i+10] == 2):
                    self.labels.append(1)
                else:
                    self.labels.append(0)
                self.ID.append(ln1[0])
            self.samples=torch.tensor(self.samples)
            self.labels=torch.tensor(self.labels)
    def __getitem__(self,idx):
        return self.samples[idx], self.labels[idx]
    def __len__(self):
        return len(self.ID)

class BCTestDataset(torch.utils.data.Dataset):
    def __init__(self, transform = None):
        f = os.path.join('./data/MNIST/bcwrapper/breast-cancer-wisconsin.data')
        ln1 = []
        self.samples = []
        self.labels = []
        self.ID = []
        with open(f, 'r') as opn:
            for l in opn:
                ln1.extend(l.strip().split(','))
            for c in range(len(ln1)): 
                if ln1[c] == '?': 
                    ln1[c] = '0'
            
            ln1 = list(map(int, ln1))
            sz = int(len(ln1)/2)
            for i in range(sz, len(ln1)-11, 11):
                tup_v = tuple(ln1[i+1:i+10])
                self.samples.append(tup_v)
                if(ln1[i+10] == 2):
                    self.labels.append(1)
                else:
                    self.labels.append(0)
                self.ID.append(ln1[0])
            
            self.samples=torch.tensor(self.samples)
            self.labels=torch.tensor(self.labels)
    def __getitem__(self,idx):
        return self.samples[idx], self.labels[idx]
    def __len__(self):
        return len(self.ID)
    
    
bc = BCTrainDataset()
bc2 = BCTestDataset()

bc_data_loader = torch.utils.data.DataLoader(bc, batch_size = 7, shuffle=True, num_workers = num_wrk)
bc_test_loader = torch.utils.data.DataLoader(bc2, batch_size=7, shuffle = True, num_workers = num_wrk)
    
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fl_1 = nn.Linear(9, 50)
        self.relu1 = torch.nn.ReLU()
        self.flt = nn.Linear(50, 30);
        self.relu2 = torch.nn.ReLU()
        self.fl3 = nn.Linear(30, 2)
    def forward(self, x):
        inp = self.fl_1(x)
        rel1 = self.relu1(inp)
        inp2 = self.flt(rel1)
        rel2 = self.relu2(inp2)
        fin = self.fl3(rel2)
        return fin
    
tst = FeedforwardNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tst.parameters(), lr=learning_rate)  

for i in range(2):
    #idx is data, elem is label assigned to data
    for j, (idx, elem) in enumerate(bc_data_loader):
        print(idx.float())
        print(elem)
        print("Epoch {}, Batch {}".format(i,j))
        res = tst(idx.float())
        #print(torch.tensor(tup[0],dtype=int))
        loss = criterion(res,elem)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


tot_size=0
num_correct=0
with torch.no_grad():
    for (idx,elem) in (bc_test_loader):
        print(idx) 
        is_correct=tst(idx.float())
        _, res = torch.max(is_correct.data, 1)
        num_correct += (res==elem).sum().item()
        tot_size+=(res.size(0))

        
print( "{} success rate on test dataset".format(num_correct/tot_size))
        

     
    
    
    


        
        
        


    
    


