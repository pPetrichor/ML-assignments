import numpy as np
from torch.utils.data import Dataset


# 定义数据集
class pendigits(Dataset):
    def __init__(self, train=True):
        super(pendigits, self).__init__()
        self.train = train

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            filename = 'pendigits.tra'
        else:
            filename = 'pendigits.tes'

        data = np.loadtxt(filename, delimiter=',', dtype=np.float32, unpack=False)

        self.datas = data[:, :16]
        self.labels = data[:, 16]

        # 数据预处理，标准化
        mean = np.mean(self.datas, axis=0)
        std = np.std(self.datas, axis=0)
        self.datas = (self.datas - mean) / std

    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.labels)