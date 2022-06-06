import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(42)

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.002)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.01, b=0.01)
        nn.init.constant_(layer.bias, 0.01)

class BadNet(nn.Module):

    def __init__(self, input_size, output_num=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 800 if input_size == 3 else 512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_num),
            # n.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    """ Model has two convolutional layers, two pooling  layers and a hidden layer with 128 units."""
    def __init__(self, input_size=3):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=48, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if input_size == 3:
            self.fc_features = 6*6*96
        else:
            self.fc_features = 5*5*96
        self.fc1 = nn.Linear(in_features=self.fc_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # reshape x
        # x = x.view(-1, self.fc_features)  # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MlleaksMLP(torch.nn.Module):
    """
    This is a simple multilayer perceptron with 64-unit hidden layer and a softmax output layer.
    """
    def __init__(self, input_size=3, hidden_size=64, output=1):
        super(MlleaksMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        output = self.sigmoid(output)
        return output
