import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device \n")

class data(Dataset):
    def __init__(self):
        
        self.path = r'C:\rois_cc400_CPAC\data\\'
        self.data = []
        folders = os.listdir(self.path)
        for f in folders:
            for i in os.listdir(self.path + f):
                self.data.append([i, f])
        self.class_map  = {'negative' : 0.0, 'positive': 1.0}
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        i, f = self.data[index]
        x = pd.read_csv(self.path + f + "\\" +  i, index_col=0)
        x = torch.tensor(x.to_numpy(), dtype=torch.float32)
        class_id = self.class_map[f]
        y = class_id
        y = torch.tensor(y)
        return torch.unsqueeze(x, 0), torch.unsqueeze(y, 0), i
    
#feedforward classifier
class cnn_ff(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.convl1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5))
        
        self.convl2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5))
        
        self.convl3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3))
        
        self.convl4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3))
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.drop1 = torch.nn.Dropout(0.25)
        
        self.lin1 = nn.Linear(15488, 1)
        
        self.bn = nn.BatchNorm2d(128)
        
        
    def forward(self, x):

        x = self.maxpool(F.leaky_relu(self.convl1(x)))
        
        x = self.maxpool(F.leaky_relu(self.convl2(x)))
        
        x = self.maxpool(F.leaky_relu(self.convl3(x)))     
        
        x = self.maxpool(F.leaky_relu(self.convl4(x)))
        
        x = self.bn(x)
           
        x = self.maxpool(x)
        
        x = torch.flatten(x, 1)
        
        x = self.drop1(x)
        
        x = self.lin1(x)
        
        return (x)
    
#concatenation classifier   
class cnn_cc(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=392, kernel_size=(1, 392))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=392, kernel_size=(3, 392))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=392, kernel_size=(5, 392))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=392, kernel_size=(7, 392))
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=392, kernel_size=(9, 392))
        
        self.pool = nn.MaxPool2d(1, 392)
        
        self.drop = torch.nn.Dropout(0.25)
        
        self.lin = nn.Linear(1960, 1)
        
    def forward(self, x):
        
        x1 = self.pool(F.leaky_relu(self.conv1(x)))         
        x2 = self.pool(F.leaky_relu(self.conv2(x)))             
        x3 = self.pool(F.leaky_relu(self.conv3(x)))    
        x4 = self.pool(F.leaky_relu(self.conv4(x))) 
        x5 = self.pool(F.leaky_relu(self.conv5(x))) 
        
        x = torch.cat((x1, x2, x3, x4, x5), 1)

        x = self.drop(x)

        x = torch.flatten(x, 1)  
        
        x = self.lin(x)
     
        return (x)

dataset = data()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

kfold = 1
kfoldacc = []
kfoldprec = []
kfoldrec = []
wrongdiag = []
rightdiag = []
losss = []

def train(epoch):
    print("training fold:", fold+1)
    for e in range(epoch):
        running_loss = 0.0

        for i, data in enumerate(data_loader_training, 0):

            inputs, labels, name = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("epoch:", e)
        print("running loss:", running_loss/i, "\n")
        losss.append(running_loss/i)
        

for fold in range(kfold):
    
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        data_loader_training = DataLoader(train_dataset, batch_size=16, shuffle=True)
        data_loader_test = DataLoader(test_dataset, batch_size=16, shuffle=True)

        dataiter_train = iter(data_loader_training)
        data, labels, name = next(dataiter_train)

        dataiter_test = iter(data_loader_test)
        data_test, labels_test, name = next(dataiter_test)
                
        model = cnn_cc().to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
                 
        train(75)
        
        model.eval() 
        
        print("evaluating fold:", fold+1)
        
        with torch.no_grad():
             correct = 0
             pos_correct = 0
             neg_correct = 0
             total = 0
             pos = 0
             neg = 0
             
             for data_test, labels, name in data_loader_test:
                 data_test = data_test.to(device)
                 diagnosis_ = []

                 
                 for label in labels:
                     diagnosis_.append(int(label))
                     if label == 1:
                         pos = pos+1
                     else:
                         neg = neg+1
                         
                         
                 outputs1 = model(data_test)                 
                 outputs2 = torch.sigmoid(outputs1)
                 output = outputs2.tolist()
                 
                 i = 0
                 
                 for z in output:
                     diag = diagnosis_[i]
                     
                     if (output[i][0] < 0.5):
                         pred = 0
                         
                         if diag == pred:
                             correct = correct+1
                             neg_correct = neg_correct+1
                            
                     elif (output[i][0] > 0.5):
                         pred = 1
                         
                         if diag == pred:
                             correct = correct+1
                             pos_correct = pos_correct+1    
                     i = i+1

                 total += labels.size(0)
                 
                 
        print('fold ', fold+1, 'accuracy:',(100 * correct / total))
                 
        if (pos_correct + (neg - neg_correct)) == 0:
            print('fold ', fold+1, 'recall: 0')
            kfoldrec.append(0)
            
        else:
            print('fold ', fold+1, 'recall:',(100 * pos_correct/(pos_correct + (neg - neg_correct))))
            kfoldrec.append(100 * (pos_correct/(pos_correct + (neg - neg_correct))))
        
        
        kfoldacc.append(100 * correct / total)
       
        
        if (pos_correct + (pos - pos_correct)) == 0:
            print('fold ', fold+1, 'precision: 0')
            kfoldprec.append(0)
            
        else:
            print('fold ', fold+1, 'precision:',(100 * pos_correct/(pos_correct + (pos - pos_correct))), "\n")
            kfoldprec.append(100 * (pos_correct)/(pos_correct + (pos - pos_correct)))
            
wd = pd.DataFrame(wrongdiag)
rd = pd.DataFrame(rightdiag)
wd.to_csv('wd_cc.csv')
rd.to_csv('rd_cc.csv')
ls = pd.DataFrame(losss)
ls.to_csv('loss_cc.csv')
            
print("mean accuracy:",np.mean(kfoldacc))
print("mean recall", np.mean(kfoldrec))
print("mean precision",np.mean(kfoldprec))

print("max accuracy:",np.max(kfoldacc))
print("max recall", np.max(kfoldrec))
print("max precision",np.max(kfoldprec))


def saliency(datatest, model):

    for param in model.parameters():
        param.requires_grad = False
    

    model.eval()

    input = datatest
    input.unsqueeze_(0)

    input.requires_grad = True

    preds = model(input)
    score, indices = torch.max(preds, 1)

    score.backward()

    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)

    return slc

x = saliency(data_test[0], model)

x = x.cpu()
pd_x = pd.DataFrame(x.numpy())
pd_x.to_csv('saliencymap_cc.csv')

plt.pcolormesh(x)
plt.colorbar()
plt.show()
