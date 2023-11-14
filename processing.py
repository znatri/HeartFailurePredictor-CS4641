# Author: Azhan Khan
# Date: 11/03/2023
# Description: This file contains the functions that are used to process the Kaggle Dataset

import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/heart.csv')
print(df.head())
print(df.dtypes)

print(df.isnull().sum())

print(df.describe())

print(df.corr())

features = df.iloc[:, :-1]
target = df.iloc[:, -1]

smote = SMOTE()
features_balanced, target_balanced = smote.fit_resample(features, target)

features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=0.15)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4, 6]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(features_train, target_train)

print("Best parameters:", grid_search.best_params_)

model = RandomForestClassifier(n_estimators=100)

model.fit(features_train, target_train)

target_predict = model.predict(features_test)

accuracy = accuracy_score(target_test, target_predict)
confusion_mat = confusion_matrix(target_test, target_predict)
class_report = classification_report(target_test, target_predict)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", confusion_mat)
print("Classification Report:\n", class_report)
