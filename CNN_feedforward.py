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
        return torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)
    
class cnn(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        def cnn_part(self, x):

            x = F.leaky_relu(self.convl1(x))
            x = F.leaky_relu(self.convl2(x))
            x = self.maxpool(F.leaky_relu(self.convl2(x)))       
                   
            x = self.maxpool(F.leaky_relu(self.convl3(x)))
                
            x = self.maxpool(F.leaky_relu(self.convl4(x)))        
                
            x = self.maxpool(F.leaky_relu(self.convl5(x)))
                
            x = torch.flatten(x, 1)
                
            x = self.lin1(x)
                
            return (x)
            
            
    def forward(self, x):

        x1 = self.cnn_part(x)
        x2 = self.cnn_part(x)
        x3 = self.cnn_part(x)
            
        x = torch.cat((x1, x2, x3), 1)
        x = self.lin1(x)
            
        return (x)
    
dataset = data()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

kfold = 1
kfoldacc = []
kfoldprec = []
kfoldrec = []

def train(epoch):
    print("training fold:", fold)
    for e in range(epoch):
        running_loss = 0.0

        for i, data in enumerate(data_loader_training, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("epoch:", e)
        print("running loss:", running_loss/i, "\n")
        

for fold in range(kfold):
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        data_loader_training = DataLoader(train_dataset, batch_size=16, shuffle=True)
        data_loader_test = DataLoader(test_dataset, batch_size=16, shuffle=True)

        dataiter_train = iter(data_loader_training)
        data, labels = next(dataiter_train)

        dataiter_test = iter(data_loader_test)
        data_test, labels_test = next(dataiter_test)
                
        model = cnn().to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
                 
        train(300)

        torch.save(model.state_dict(), r'C:\rois_cc400_CPAC\cnn_state' + str(kfold) + '.pt')
        
        model.eval() 
        
        print("evaluating fold:", fold+1)
        
        with torch.no_grad():
             correct = 0
             pos_correct = 0
             neg_correct = 0
             total = 0
             pos = 0
             neg = 0
             
             for data_test, labels in data_loader_test:
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

x = saliency(data_test[1], model)

x = x.cpu()

plt.imshow(x, cmap=plt.cm.hot)
plt.axis('off')
plt.show()
