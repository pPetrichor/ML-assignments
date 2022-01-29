from sklearn import svm
from imblearn.combine import SMOTEENN
from collections import Counter
import clean
from data_utils import *
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

train_data = np.loadtxt('anonymous.tra', dtype=int, delimiter=',')
train_inputs, train_targets = train_data[:, :-1], train_data[:, -1]

new_train_inputs, new_train_targets = [], []
eps = [300, 3000, 300, 500, 100, 1000, 200, 100, 400, 900]

for i in range(10):
    index = np.argwhere(train_targets == i)
    data = train_inputs[index].reshape(-1, 64)
    clean_data_i = clean.clean_i(data, eps[i], 64)

    for j in range(len(clean_data_i)):
        new_train_inputs.append(clean_data_i[j])
        new_train_targets.append(i)

new_train_inputs = np.array(new_train_inputs)
# print(new_train_inputs.shape)
new_train_targets = np.array(new_train_targets)
# print(Counter(train_targets))
# print(Counter(new_train_targets))

oversample = SMOTEENN(random_state=217)

in_oversampled, tar_oversampled = oversample.fit_resample(new_train_inputs, new_train_targets)

SVM = svm.SVC(gamma='scale', C=0.8, decision_function_shape='ovr', kernel='rbf')
SVM.fit(in_oversampled, tar_oversampled)

test_inputs = np.loadtxt('anonymous.tes', dtype=int, delimiter=',')
res_pred = SVM.predict(test_inputs)
result = '\n'.join(str(i) for i in res_pred)

with open('output_181860092.txt', 'w') as f:
    f.write(result)


