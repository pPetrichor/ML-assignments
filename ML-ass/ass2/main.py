# import utils
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch import optim
# from torch.autograd import Variable
# import pendigits
# import models_
#
# # 加载数据集
# train_dataset = pendigits.pendigits(train=True)
# test_dataset = pendigits.pendigits(train=False)
#
# train_loader_64 = DataLoader(dataset=train_dataset, batch_size=64)
# test_loader = DataLoader(dataset=test_dataset, batch_size=64)
#
# train_loader_16 = DataLoader(dataset=train_dataset, batch_size=16)
#
# train_loader_256 = DataLoader(dataset=train_dataset, batch_size=256)
#
#
# models = [models_.Net()]#[models_.per(), models_.MLP(), models_.Net()]
# learning_rates = [0.1, 0.05, 0.01, 3e-4]
# loss_functions = ['mse', 'cross_entropy']
# train_loaders = [train_loader_64, train_loader_16, train_loader_256]
#
#
# def train(model, optimizer, train_loss, loss_function, train_loader):
#     model.train()
#     for idx, (data, label) in enumerate(train_loader):
#         data, label = Variable(data), Variable(label)
#
#         optimizer.zero_grad()  # 梯度清零
#
#         output = model(data)
#
#         if loss_function == 'mse':
#             loss = F.mse_loss(F.softmax(output, dim=1), utils.one_hot(label.long()))
#
#         elif loss_function == 'cross_entropy':
#             loss = F.cross_entropy(output, label.long())
#
#         loss.backward()
#
#         optimizer.step()
#
#         train_loss.append(loss.item())
#
#         # if idx % 50 == 0:
#         #     print("epoch: ", epoch, "batch_idx: ", idx, "loss: ", loss.item())
#
#
# def test(model):
#     model.eval()
#     total_correct = 0
#     for data, label in test_loader:
#         data, label = Variable(data), Variable(label)
#         output = model(data)
#         pred = output.argmax(dim=1)  # 预期结果
#         num_correct = pred.eq(label).sum().float().item()
#         total_correct += num_correct
#
#     acc = total_correct / len(test_loader.dataset)
#     print('测试集准确率:', acc)
#
#
# train_losses = []
# label_names = []
# for loss_function in loss_functions:
#     for lr in learning_rates:
#         for train_loader in train_loaders:
#             for model in models:
#                 train_losses.clear()
#                 label_names.clear()
#
#                 optimizers = \
#                     {'sgd': optim.SGD(model.parameters(), lr=lr),
#                      'momentum': optim.SGD(model.parameters(), lr=lr, momentum=0.8),
#                     'adam': optim.Adam(model.parameters(), lr=lr)}
#                 for name, optimizer in optimizers.items():
#                     print('损失函数', loss_function, '学习率', lr, 'batch_size', train_loader.batch_size, '当前模型', model.getname(), '优化器', name)
#                     plt_name = loss_function + '+' + str(train_loader.batch_size)
#
#                     model.weight_init()  # 重新初始化模型的权重参数
#
#                     train_loss = []
#                     for epoch in range(5):
#                         train(model, optimizer, train_loss, loss_function, train_loader)
#
#                     test(model)
#                     train_losses.append(train_loss)
#                     label_names.append(model.getname() + ' ' + str(lr) + ' ' + name)
#
#                 # utils.plot_curves(train_losses, label_names, plt_name)

import torch as th

a = th.tensor([[1,2,3,4,5],[5,4,3,2,1]])
b = th.tensor([[2,3,4,5,6],[6,5,4,3,2],[1,2,3,4,5],[3,2,6,1,1]])
x = th.matmul(a, b.t())

x = x.view(a.shape[0], a.shape[0], -1)
print(x.shape)
print(x)
nominator = th.eye(x.shape[0])[:, :, None]
print(nominator)
print(nominator.shape)
out = x * nominator
print(out)

