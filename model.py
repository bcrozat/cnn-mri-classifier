# Import dependencies
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# ResNet50 model
weights = ResNet50_Weights.DEFAULT
RN50 = resnet50(weights=weights)
# Modify the last layer for binary classification
RN50.fc = nn.Sequential(
    in_features=2048,
    out_features=1
)

class CNN(nn.Module): # V5
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) # 3 input channels (RGB), 32 output channels, kernel size 5x5
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # 32 input channels, 64 output channels
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(128*8*8, 1)   # Output a single value for binary classification # Use (256*4*4) for 5 convolutional layers, (128*8*8) for 4, (64*16*16) for 3
        self.elu = nn.ELU() # Exponential Linear Unit activation functio
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 2x2 max pooling layer with stride 2 which reduces the spatial dimensions by half
        self.dropout = nn.Dropout(p=0.1) # Dropout layer with 20% probability of zeroing out some elemets to prevent overfitting

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.pool(x)
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.elu(x)
        # x = self.pool(x)
        x = self.dropout(x)
        bs, _, _, _ = x.shape
        x = x.view(bs, -1)  # Flatten the tensor to feed it into the fully connected layer
        x = self.fc1(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation
        return x.squeeze() # Remove the singleton dimension (so that model output size <-> evaluation input size (torch.Size([64, 1])) which matches target (label) size (torch.Size([64]))