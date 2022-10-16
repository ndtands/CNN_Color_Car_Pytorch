from black import out
import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, dropout: float, num_classes: int):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64,kernel_size=3,stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*4*4,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.dropout(self.batchnorm2(self.pool(x)))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))
        x = x.view(-1, 64*4*4) 
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim = 1)
        return x

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

# link paper: https://arxiv.org/pdf/1510.07391.pdf
class ColorNet(nn.Module):
    def __init__(self, num_classes: int):
        super(ColorNet, self).__init__()
        # ========================= TOP BRANCH =========================
        #STAGE 1
        self.top_conv1 = nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding='valid')
        self.top_batchnorm1 = nn.BatchNorm2d(num_features=48)
        self.top_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        #STAGE 2
        self.top_top_conv2 = nn.Conv2d(in_channels=24,out_channels=64,kernel_size=3,stride=1,padding='same')
        self.top_top_batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.top_top_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.top_bot_conv2 = nn.Conv2d(in_channels=24,out_channels=64,kernel_size=3,stride=1,padding='same')
        self.top_bot_batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.top_bot_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        #STAGE 3
        self.top_conv3 = nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding='same')

        # STAGE 4
        self.top_top_conv4 = nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding='same')
        self.top_bot_conv4 = nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding='same')

        # STAGE 5
        self.top_top_conv5 = nn.Conv2d(in_channels=96,out_channels=64,kernel_size=3,stride=1,padding='same')
        self.top_top_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.top_bot_conv5 = nn.Conv2d(in_channels=96,out_channels=64,kernel_size=3,stride=1,padding='same')
        self.top_bot_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # ========================= BOTTOM BRANCH =========================
        #STAGE 1
        self.bottom_conv1 = nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding='valid')
        self.bottom_batchnorm1 = nn.BatchNorm2d(num_features=48)
        self.bottom_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        #STAGE 2
        self.bottom_top_conv2 = nn.Conv2d(in_channels=24,out_channels=64,kernel_size=3,stride=1,padding='same')
        self.bottom_top_batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.bottom_top_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.bottom_bot_conv2 = nn.Conv2d(in_channels=24,out_channels=64,kernel_size=3,stride=1,padding='same')
        self.bottom_bot_batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.bottom_bot_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        #STAGE 3
        self.bottom_conv3 = nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding='same')

        # STAGE 4
        self.bottom_top_conv4 = nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding='same')
        self.bottom_bot_conv4 = nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding='same')

        # STAGE 5
        self.bottom_top_conv5 = nn.Conv2d(in_channels=96,out_channels=64,kernel_size=3,stride=1,padding='same')
        self.bottom_top_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.bottom_bot_conv5 = nn.Conv2d(in_channels=96,out_channels=64,kernel_size=3,stride=1,padding='same')
        self.bottom_bot_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # ========================= CONCATENATE TOP AND BOTTOM BRANCH =========================
        self.FC1 = nn.Linear(in_features=21120, out_features=4096)
        self.dropout1 = nn.Dropout2d(p=0.6)
        self.FC2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = nn.Dropout2d(p=0.6)
        self.FC3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        # ========================= TOP BRANCH =========================
        #STAGE 1
        top_conv1 = self.top_batchnorm1(F.relu(self.top_conv1(x)))
        top_conv1 = self.top_pool1(top_conv1)

        # STAGE 2
        top_top_conv2 = LambdaLayer(lambda x : x[:,:24,:,:])(top_conv1)
        top_bot_conv2 = LambdaLayer(lambda x : x[:,24:,:,:])(top_conv1)

        top_top_conv2 = self.top_top_batchnorm2(F.relu(self.top_top_conv2(top_top_conv2)))
        top_top_conv2 = self.top_top_pool2(top_top_conv2)

        top_bot_conv2 = self.top_bot_batchnorm2(F.relu(self.top_bot_conv2(top_bot_conv2)))
        top_bot_conv2 = self.top_bot_pool2(top_bot_conv2)

        # STAGE 3
        top_conv3 = torch.cat((top_top_conv2, top_bot_conv2), -1)
        top_conv3 = self.top_conv3(top_conv3)

        # STAGE 4
        top_top_conv4 = LambdaLayer(lambda x : x[:,:96,:,:])(top_conv3)
        top_bot_conv4 = LambdaLayer(lambda x : x[:,96:,:,:])(top_conv3)

        top_top_conv4 = F.relu(self.top_top_conv4(top_top_conv4))
        top_bot_conv4 = F.relu(self.top_bot_conv4(top_bot_conv4))

        # STAGE 5
        top_top_conv5 = self.top_top_pool5(F.relu(self.top_top_conv4(top_top_conv4)))
        top_bot_conv5 = self.top_bot_pool5(F.relu(self.top_bot_conv4(top_bot_conv4)))

        # ========================= TOP BRANCH =========================
        #STAGE 1
        bottom_conv1 = self.bottom_batchnorm1(F.relu(self.bottom_conv1(x)))
        bottom_conv1 = self.bottom_pool1(bottom_conv1)

        # STAGE 2
        bottom_top_conv2 = LambdaLayer(lambda x : x[:,:24,:,:])(bottom_conv1)
        bottom_bot_conv2 = LambdaLayer(lambda x : x[:,24:,:,:])(bottom_conv1)

        bottom_top_conv2 = self.bottom_top_batchnorm2(F.relu(self.bottom_top_conv2(bottom_top_conv2)))
        bottom_top_conv2 = self.bottom_top_pool2(bottom_top_conv2)

        bottom_bot_conv2 = self.bottom_bot_batchnorm2(F.relu(self.bottom_bot_conv2(bottom_bot_conv2)))
        bottom_bot_conv2 = self.bottom_bot_pool2(bottom_bot_conv2)

        # STAGE 3
        bottom_conv3 = torch.cat((bottom_top_conv2, bottom_bot_conv2), -1)
        bottom_conv3 = self.bottom_conv3(bottom_conv3)

        # STAGE 4
        bottom_top_conv4 = LambdaLayer(lambda x : x[:,:96,:,:])(bottom_conv3)
        bottom_bot_conv4 = LambdaLayer(lambda x : x[:,96:,:,:])(bottom_conv3)

        bottom_top_conv4 = F.relu(self.bottom_top_conv4(bottom_top_conv4))
        bottom_bot_conv4 = F.relu(self.bottom_bot_conv4(bottom_bot_conv4))

        # STAGE 5
        bottom_top_conv5 = self.bottom_top_pool5(F.relu(self.bottom_top_conv4(bottom_top_conv4)))
        bottom_bot_conv5 = self.bottom_bot_pool5(F.relu(self.bottom_bot_conv4(bottom_bot_conv4)))

        # ========================= CONCATENATE TOP AND BOTTOM BRANCH =========================
        conv_output = torch.cat((top_top_conv5,top_bot_conv5,bottom_top_conv5,bottom_bot_conv5),-1)
        flatten = conv_output.view(-1, 21120)
        FC1 = self.dropout1(F.relu(self.FC1(flatten)))
        FC2 = self.dropout2(F.relu(self.FC2(FC1)))
        output = self.FC3(FC2)
        return output

        


       

