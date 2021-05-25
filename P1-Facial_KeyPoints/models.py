## TODO : NaimishNet  
## https://arxiv.org/pdf/1710.00977.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # ------------------- 1 st Layer -------------------
        # Convolutional Layer - 1
        # (W - F)/S + 1 = (224 - 5)/1 + 1 = 220
        # Output Size = (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # Batch Normalization 
        self.bn1 = nn.BatchNorm2d(32)
        
        # Maxpool Layer (Pool Kernel size = 2, stride = 2)
        # Output Size = (32, 110, 110)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Dropout layer .4 
        self.fc_drop1 = nn.Dropout(p=0.4)
        
        # ------------------- 2 nd Layer -------------------
        # Convolutional Layer - 2
        # (W - F)/S + 1 = (110 - 5)/1 + 1 = 106
        # Output Size = (64, 106, 106)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # Batch Normalization
        self.bn2 = nn.BatchNorm2d(64)
        
        # Maxpool Layer (Pool Kernel size = 2, stride = 2)
        # Output Size = (64, 53, 53)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Dropout Layer .4
        self.fc_drop2 = nn.Dropout(p=0.4)
        
        # ------------------- 3 rd Layer -------------------
        # Convolutional Layer - 3
        # (W - F)/S + 1 = (53 - 5)/1 + 1
        # Output Size (128, 49, 49)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        # Batch Normalization 
        self.bn3 = nn.BatchNorm2d(128)
        
        # Maxpool Layer (Pool Kernel size = 2, stride = 2)
        # Output Size = (128, 24, 24) -- Round Down 
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Dropout Layer .4
        self.fc_drop3 = nn.Dropout(p=0.4)
        
        # ------------------- 4 th Layer -------------------
         # Convolutional Layer - 4
        # (W - F)/S + 1 = (24 - 5)/1 + 1
        # Output Size (256, 20, 20)
        self.conv4 = nn.Conv2d(128, 64, 5)
        
        # Batch Normalization
        self.bn4 = nn.BatchNorm2d(64)
        
        # Maxpool Layer (Pool Kernerl size = 2, stride = 2)
        # Output Size = (64, 10, 10)
        self.pool4 = nn.MaxPool2d(2,2)
        
        # Dropout Layer .4
        self.fc_drop4 = nn.Dropout(p=0.4)
        
        # -- Fully Connected layer --
        self.fc5 = nn.Linear(64*10*10, 136)

        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.fc_drop1(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.fc_drop2(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.fc_drop3(x)
        
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.fc_drop4(x)
        
        # Flatten (Convert Image into Vector Representation)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        
        
        return x
