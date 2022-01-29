import torch
from matplotlib import pyplot as plt

colors = ['r', 'g', 'b']


def one_hot(label, nums=10):  # 将label转换为one-hot编码
    ret = torch.zeros(label.size(0), nums)
    ret.scatter_(1, torch.LongTensor(label).view(-1, 1), 1)
    return ret


def plot_curve(loss, curve_label, color):  # 绘制曲线图
    plt.plot(range(len(loss)), loss, color=color, label=curve_label)


def plot_curves(losses, curve_labels, plt_name):
    for i in range(len(losses)):
        plot_curve(losses[i], curve_labels[i], colors[i%3])

    plt.legend()
    plt.title(plt_name)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()
