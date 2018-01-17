# predictive model using linear regression

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
# from sklearn.datasets import make_regression

#
# Training data
df_training = pd.read_csv('C:\\Users\\rivas\\OneDrive\\Documents\\JMR\\Education\\Springboard\\Projects\\Capstone1\\fashionmnisttrain.csv')
x_train = df_training.iloc[:, 1:]
y_train = df_training.iloc[:, :1]

#

#
# Test data
df_test = pd.read_csv('C:\\Users\\rivas\\OneDrive\\Documents\\JMR\\Education\\Springboard\\Projects\\Capstone1\\fashionmnisttest.csv')
x_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:, :1]

#
# initialize our regressor object
reg = LinearRegression()

# train the model
reg.fit(x_train, y_train)

# betas
reg.coef_[:10,:10]

#Predict on the test data
y_test_predicted = reg.predict(x_test)

y_test_predicted[:10,:10]
y_test.iloc[:10]

mse = ((y_test - y_test_predicted)**2).sum()
mse

#
#







