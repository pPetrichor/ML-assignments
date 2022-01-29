from torch import nn
import torch.nn.functional as F


# 多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def getname(self):
        return 'MLP'

    def weight_init(self):  # 重新初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# 神经网络模型(增加层数并引入BN与Dropout)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, 10)

        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.bn2 = nn.BatchNorm1d(num_features=12)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc3(x)

    def getname(self):
        return 'Net'

    def weight_init(self):  # 重新初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)