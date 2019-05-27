# linear svm

from deal_data import data
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

x_train = data()[0]
x_test = data()[1]
y_train = data()[2]
y_test = data()[3]

def fit_grid_point_Linear(C, x_train, y_train, x_test, y_test):
    svc = LinearSVC(C=C)
    svc = svc.fit(x_train, y_train)
    accuracy = svc.score(x_test, y_test)
    return accuracy

def fit_grid_point_RBF(C, gamma, x_train, y_train, x_test, y_test):
    svc = SVC(C=C, kernel='rbf', gamma=gamma)
    svc = svc.fit(x_train, y_train)
    accuracy = svc.score(x_test, y_test)
    return accuracy

Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
accuracy = []
# for i in range(len(Cs)):
#     tmp = fit_grid_point_Linear(Cs[i], x_train, y_train, x_test, y_test)
#     accuracy.append(tmp)
# print(accuracy)
# figure = plt.figure()
# x = np.log(Cs)
# y = accuracy
# plt.plot(x, y)
# plt.show()

for i in range(len(Cs)):
    for j in range(len(gamma)):
        tmp = fit_grid_point_RBF(Cs[i], gamma[j], x_train, y_train, x_test, y_test)
        accuracy.append(tmp)
print(accuracy)
figure = plt.figure()
accuracy = np.array(accuracy).reshape(len(Cs), len(gamma))
print(accuracy)
x = np.log(Cs)
for i in range(len(gamma)):
    plt.plot(x, np.array(accuracy[:, i]), label = 'log(gamma)' + str(np.log(gamma)))
plt.show()


