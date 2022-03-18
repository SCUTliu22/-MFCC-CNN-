import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(16,2),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d((2,1), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(8,2),
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(4,2),
                stride=1,
            ),
            nn.ReLU(),
        )
        self.input_layer = nn.Linear(1280, 1024)
        self.layer_output = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        # [b,1,199,13]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_output(x)
        return x