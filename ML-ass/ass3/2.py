from sklearn import tree
from data_utils import *
import model
import pendigits
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import time


def train(model, train_loader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    for idx, (data, label) in enumerate(train_loader):
        data, label = Variable(data), Variable(label)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)
        loss = F.cross_entropy(output, label.long())
        loss.backward()
        optimizer.step()


def test(model):
    model.eval()
    total_correct = 0

    for data, label in test_loader:
        data, label = Variable(data), Variable(label)
        output = model(data)
        pred = output.argmax(dim=1)  # 预期结果
        num_correct = pred.eq(label).sum().float().item()
        total_correct += num_correct

    acc = total_correct / len(test_loader.dataset)
    return acc


imb_ratios = [1, 10, 100]
noise_levels = [0, 0.1, 0.3, 0.4, 0.7]
test_inputs, test_targets = get_data(train=False)
test_dataset = pendigits.pendigits(inputs=test_inputs, targets=test_targets)
test_loader = DataLoader(dataset=test_dataset, batch_size=64)

for i in imb_ratios:
    for n in noise_levels:
        print(i, n)
        train_inputs, train_targets = get_data(train=True, corrupt=True, imb_ratio=i, noise_level=n)

        clf = tree.DecisionTreeClassifier(criterion="entropy")
        start = time.perf_counter()
        clf = clf.fit(train_inputs, train_targets)
        end = time.perf_counter()
        acc_clf = clf.score(test_inputs, test_targets)
        print('决策树: ', acc_clf, '训练时间: ', end-start)

        rfc = RandomForestClassifier(criterion="entropy", n_estimators=25)
        start = time.perf_counter()
        rfc = rfc.fit(train_inputs, train_targets)
        end = time.perf_counter()
        acc_rfc = rfc.score(test_inputs, test_targets)
        print('随机森林: ', acc_rfc, '训练时间: ', end-start)

        SVM = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
        start = time.perf_counter()
        SVM.fit(train_inputs, train_targets)
        end = time.perf_counter()
        acc_SVM = SVM.score(test_inputs, test_targets)
        print('SVM : ', acc_SVM, '训练时间: ', end-start)

        train_dataset = pendigits.pendigits(inputs=train_inputs, targets=train_targets)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64)

        mlp = model.MLP()

        start = time.perf_counter()
        for epoch in range(5):
            train(mlp, train_loader)

        end = time.perf_counter()
        acc_mlp = test(mlp)
        print('mlp准确率:', acc_mlp, '训练时间: ', end-start)
        print('-----------------------')