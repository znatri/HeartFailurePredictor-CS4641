# Author: Azhan Khan
# Date: 11/03/2023
# Description: This file contains the functions that are used to process the Kaggle Dataset

import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy

df = pd.read_csv('data/heart.csv')
print(df.head())
print(df.dtypes)

print(df.isnull().sum())

print(df.describe())

print(df.corr())

# Outlier handling for selected numerical features
numerical_features = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

for feature in numerical_features:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    df[feature] = np.where(df[feature] <= q1, q1, df[feature])
    df[feature] = np.where(df[feature] >= q3, q3, df[feature])

# Label Encoding
lbl = LabelEncoder()
df['Sex'] = lbl.fit_transform(df['Sex'])
df['ST_Slope'] = lbl.fit_transform(df['ST_Slope'])

# One-Hot Encoding
encoder = OneHotEncoder()
df_encoded = pd.DataFrame(encoder.fit_transform(df[['RestingECG']]).toarray(), columns=encoder.get_feature_names_out(['RestingECG']))
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(['RestingECG'], axis=1)

# Split the data into features and target
features = df.iloc[:, :-1]
target = df.iloc[:, -1]

# Handling class imbalance using SMOTE
smote = SMOTE()
features_balanced, target_balanced = smote.fit_resample(features, target)

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features_balanced, target_balanced, test_size=0.15)

# Standardize the features
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4, 6]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(features_train, target_train)

print("Best parameters:", grid_search.best_params_)

# Train the model with the best hyperparameters
model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                               max_depth=grid_search.best_params_['max_depth'],
                               min_samples_split=grid_search.best_params_['min_samples_split'])

model.fit(features_train, target_train)

# Make predictions on the test set
target_predict = model.predict(features_test)

# Evaluate the model performance
accuracy = accuracy_score(target_test, target_predict)
confusion_mat = confusion_matrix(target_test, target_predict)
class_report = classification_report(target_test, target_predict)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", confusion_mat)
print("Classification Report:\n", class_report)
