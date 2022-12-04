import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# %matplotlibe inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


advert = pd.read_csv('Advertising.csv')
# print(advert.head()) - printing 5 raw data from up
# print(advert.info()) - showing information about specific data - memory, columns, class and etc.
# print(advert.columns) - showing name of the columns in the data

advert.drop(['Unnamed: 0'], axis=1, inplace=True)


# advert.drop(['Unnamed: 0'], axis=1, inplace=True)) - removing the column and saving the change in data
# print(advert.head()) - checking the result of dropping

# sns.histplot(advert.sales)
# sns.histplot(advert.newspaper)
# sns.histplot(advert.radio)
# sns.histplot(advert.TV)
# plt.show() - showing the  visualization of the data

# sns.pairplot(advert, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=7, aspect=0.7, kind='reg')
# plt.show() - exploring relationships between predictors and response

# print(advert.TV.corr(advert.sales)) - showing TV correlation
# print(advert.corr()) - correlation of data

# sns.heatmap(advert.corr(), annot=True) - visualisation of the correlation
# plt.show()

X = advert[['TV']]
# print(X.head())

# print(type(X))
# print(X.shape)

y = advert.sales
# print(type(y))
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

# print(linreg.intercept_)
# print(linreg.coef_)

y_pred = linreg.predict(X_test)
# print(y_pred[:5])

true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]

# print((10 + 0 + 20 + 10) / 4)
# print(metrics.mean_absolute_error(true, pred)) - mean absolute error

# print((10**2 + 0**2 + 20**2 + 10**2) / 4)
# print(metrics.mean_squared_error(true, pred)) - mean squared error

# print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2) / 4))
# print(np.sqrt(metrics.mean_squared_error(true, pred))) - root mean squared error


# print(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
