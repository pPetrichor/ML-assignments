from sklearn import svm
from imblearn.combine import SMOTEENN
from collections import Counter
import clean
from data_utils import *

test_inputs, test_targets = get_data(train=False)
train_inputs, train_targets = get_data(train=True, corrupt=True, imb_ratio=10, noise_level=0.3)

new_train_inputs, new_train_targets = [], []
eps = [100, 100, 50, 50, 30, 30, 10, 10, 10, 10]
for i in range(10):
    index = np.argwhere(train_targets == i)
    data = train_inputs[index].reshape(-1, 16)
    clean_data_i = clean.clean_i(data, eps[i], 16)

    for j in range(len(clean_data_i)):
        new_train_inputs.append(clean_data_i[j])
        new_train_targets.append(i)

new_train_inputs = np.array(new_train_inputs)
# print(new_train_inputs.shape)
new_train_targets = np.array(new_train_targets)
print(Counter(new_train_targets))

oversample = SMOTEENN(random_state=217)
# X_train, X_val, y_train, y_val = train_test_split(np.array(new_train_inputs), np.array(new_train_targets), test_size=0.3, random_state=217)

# in_oversampled, tar_oversampled = oversample.fit_resample(X_train, y_train)
in_oversampled, tar_oversampled = oversample.fit_resample(new_train_inputs, new_train_targets)
SVM = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
SVM.fit(in_oversampled, tar_oversampled)
acc_SVM = SVM.score(test_inputs, test_targets)
print('SVM : ', acc_SVM)
