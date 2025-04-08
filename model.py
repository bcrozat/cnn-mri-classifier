# Import dependencies
import torch
import torch.nn as nn

class CNN(nn.Module): # V5
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) # 3 input channels (RGB), 32 output channels, kernel size 5x5
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 32 input channels, 64 output channels
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        #self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm2d(num_features=64)
        #self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.bn5 = nn.BatchNorm2d(num_features=64)

        self.fc1 = nn.Linear(64*128*128, 1)   # Output a single value for binary classification
        self.elu = nn.ELU() # Exponential Linear Unit activation functio
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 2x2 max pooling layer with stride 2 which reduces the spatial dimensions by half

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        #x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        #x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        #x = self.pool(x)
        bs, _, _, _ = x.shape
        x = x.view(bs, -1)  # Flatten the tensor to feed it into the fully connected layer
        x = self.fc1(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation
        return x.squeeze() # Remove the singleton dimension (so that model output size <-> evaluation input size (torch.Size([64, 1])) which matches target (label) size (torch.Size([64]))

# class CNNModelGPT(nn.Module):
#     def __init__(self):
#         super(CNNModelGPT, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, 5)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(256, 1)  # Output a single value for binary classification
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         bs, _, _, _ = x.shape
#         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
#         x = self.dropout(x)
#         x = torch.sigmoid(self.fc1(x))  # Apply sigmoid activation
#         return x