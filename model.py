import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, H,W,in_channels=1, num_classes=10,):
        super(CNN, self).__init__()
        self.out_channels=8
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=6, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(int(H*W*self.out_channels/16), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x