# Author: Nicholas Arribasplata
# Date: 12/03/2023
# Description: This file contains the linear regression model for our dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart.csv')
print(df.head())
print(df.dtypes)
print(df.isnull().sum())

features = df.iloc[:, :-1]
target = df.iloc[:, -1]

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.15, random_state=42)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

linear_model = LinearRegression()

linear_model.fit(features_train, target_train)

target_predict = linear_model.predict(features_test)

mse = mean_squared_error(target_test, target_predict)
r2 = r2_score(target_test, target_predict)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')