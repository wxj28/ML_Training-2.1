# deal with data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
def data():
    data = pd.read_csv('diabetes.csv')
    # print(data.info())
    # print(data.describe())
    # print(data.head())
    # print(data.isnull().sum())
    # figure = plt.figure()
    # data.hist()
    # sns.countplot(data.Outcome)
    # plt.show()
    # corr = data[data.columns].corr()
    # sns.heatmap(corr, annot=True)
    # plt.show()

    # select feature
    x = data.iloc[:, 0:8]
    y = data.iloc[:, 8]
    select_top_4 = SelectKBest(score_func=chi2, k=4)
    features = select_top_4.fit(x, y)
    features = features.transform(x)
    # print(data.head())
    # print(features)
    x_features = pd.DataFrame(data=features, columns=['Glucose', 'Insulin', 'BMI', 'Age'])
    y = pd.DataFrame(data=y, columns=['Outcome'])

    # standardization

    x_scale = StandardScaler().fit_transform(x_features)
    x = pd.DataFrame(data=x_scale, columns=x_features.columns)

    # split test and train dataset

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


data()
