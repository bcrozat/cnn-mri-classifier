import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 5)

        self.fc1 = nn.Linear(256, 1)   # Output a single value for binary classification

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = torch.sigmoid(self.fc1(x))  # Apply sigmoid activation
        return x.squeeze() # Remove the singleton dimension (so that model output size <-> evaluation input size (torch.Size([64, 1])) which matches target (label) size (torch.Size([64]))

# class ImprovedCNNModel(nn.Module):
#     def __init__(self):
#         super(ImprovedCNNModel, self).__init__()
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