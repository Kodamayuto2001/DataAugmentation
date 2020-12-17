import torch 
import torch.nn.functional as F

class Net_log_softmax(torch.nn.Module):
    def __init__(self,num,inputSize,Neuron):
        super(Net_log_softmax,self).__init__()
        self.iSize = inputSize
        self.fc1 = torch.nn.Linear(self.iSize*self.iSize,Neuron)
        self.fc2 = torch.nn.Linear(Neuron,num)

    def forward(self,x):
        x = x.view(-1,self.iSize*self.iSize)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


class CNN(torch.nn.Module):
    def __init__(self,num,inputSize,hidden1,hidden2):
        super(CNN,self).__init__()
        self.iSize  = inputSize
        self.conv1  = torch.nn.Conv2d(1,4,3)
        self.bn1    = torch.nn.BatchNorm2d(4)
        self.pool   = torch.nn.MaxPool2d(2,2)
        self.conv2  = torch.nn.Conv2d(4,16,3)
        self.bn2    = torch.nn.BatchNorm2d(16)
        self.fc1    = torch.nn.Linear(16*38*38,hidden1)
        self.fc2    = torch.nn.Linear(hidden1,hidden2)
        self.fc3    = torch.nn.Linear(hidden2,num)

    def forward(self,x):
        x = self.conv1(x)
        # print(x.size())
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.pool(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.pool(x)    
        # print(x.size())
        x = x.view(-1,16*38*38)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

