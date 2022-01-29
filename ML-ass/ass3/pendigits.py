import numpy as np
from torch.utils.data import Dataset


# 定义数据集
class pendigits(Dataset):
    def __init__(self, inputs=None, targets=None):
        super(pendigits, self).__init__()

        self.datas = inputs.astype(np.float32)
        self.labels = targets.astype(np.float32)

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