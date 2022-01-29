from data_utils import *
from sklearn import tree
import model
import pendigits
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import svm
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN


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


imb_ratios = [10, 50, 100]
test_inputs, test_targets = get_data(train=False)
test_dataset = pendigits.pendigits(inputs=test_inputs, targets=test_targets)
test_loader = DataLoader(dataset=test_dataset, batch_size=64)

print('不进行过采样：')
for i in imb_ratios:  # 没有过采样
    print('imb_ratio:', i)
    train_inputs, train_targets = get_data(train=True, corrupt=True, imb_ratio=i, noise_level=0)

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(train_inputs, train_targets)
    acc_clf = clf.score(test_inputs, test_targets)
    print('决策树: ', acc_clf)

    SVM = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    SVM.fit(train_inputs, train_targets)
    acc_SVM = SVM.score(test_inputs, test_targets)
    print('SVM : ', acc_SVM)

    train_dataset = pendigits.pendigits(inputs=train_inputs, targets=train_targets)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64)

    mlp = model.MLP()

    for epoch in range(60):
        train(mlp, train_loader)

    acc_mlp = test(mlp)
    print('mlp准确率:', acc_mlp)
    print('-----------------------')

print('进行过采样：')
for i in imb_ratios:  # 进行过采样
    print('imb_ratio:', i)
    train_inputs, train_targets = get_data(train=True, corrupt=True, imb_ratio=i, noise_level=0)
    # print(Counter(train_targets))
    oversample = RandomOverSampler(random_state=217)
    in_oversampled, tar_oversampled = oversample.fit_resample(train_inputs, train_targets)
    # print(Counter(tar_oversampled))

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(in_oversampled, tar_oversampled)
    acc_clf = clf.score(test_inputs, test_targets)
    print('决策树: ', acc_clf)

    SVM = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    SVM.fit(in_oversampled, tar_oversampled)
    acc_SVM = SVM.score(test_inputs, test_targets)
    print('SVM : ', acc_SVM)

    train_dataset = pendigits.pendigits(inputs=in_oversampled, targets=tar_oversampled)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64)

    mlp = model.MLP()

    for epoch in range(60):
        train(mlp, train_loader)

    acc_mlp = test(mlp)
    print('mlp准确率:', acc_mlp)
    print('-----------------------')

print('过采样+松弛SVM：')
for i in imb_ratios:
    print('imb_ratio:', i)
    train_inputs, train_targets = get_data(train=True, corrupt=True, imb_ratio=i, noise_level=0.3)
    oversample = SMOTEENN(random_state=217)
    in_oversampled, tar_oversampled = oversample.fit_resample(train_inputs, train_targets)
    SVM = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    SVM.fit(in_oversampled, tar_oversampled)
    acc_SVM = SVM.score(test_inputs, test_targets)
    print('SVM : ', acc_SVM)

    SVM = svm.SVC(gamma='scale', C=0.7, decision_function_shape='ovr', kernel='rbf')
    SVM.fit(in_oversampled, tar_oversampled)
    acc_SVM = SVM.score(test_inputs, test_targets)
    print('松弛SVM : ', acc_SVM)
