from deal_data import data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss

x_train = data()[0]
x_test = data()[1]
y_train = data()[2]
y_test = data()[3]

# 调参

penalties = ['l1', 'l2']
Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
tuned_parameters = dict(penalty = penalties, C = Cs)
lr = LogisticRegression()
loss = cross_val_score(lr, x_train, y_train, cv=5, scoring='neg_log_loss')
grid = GridSearchCV(lr, tuned_parameters, cv=5, scoring='neg_log_loss')
grid.fit(x_train, y_train)
# print(grid.cv_results_)
print(-loss)
print(-loss.mean())
print(-grid.best_score_)
print(grid.best_params_)
print('The best regular function is : ' + grid.best_params_['penalty'])
print('The best regular parameter is : ' + str(grid.best_params_['C']))
y = lr.predict(x_test, penalties)