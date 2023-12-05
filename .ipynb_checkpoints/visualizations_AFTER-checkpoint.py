# Author: Felipe Mairhofer, Eric Vela, Hardik Goel
# Date: 11/03/2023
# Description: This file contains the functions that are used to process the Kaggle Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy
import seaborn as sns

df = pd.read_csv('data/heart.csv')
print(df.head())
print(df.dtypes)

print(df.isnull().sum())

print(df.describe())

print(df.corr())

# Outlier handling for selected numerical features
numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

for feature in numerical_features:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
    df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])

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


feature_importances = model.feature_importances_
feature_names = features.columns
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Classifier - Feature Importances')
plt.show()

labels = np.unique(target)
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


train_accuracy = []
test_accuracy = []
num_trees_range = range(1, 101)

for num_trees in num_trees_range:
    model = RandomForestClassifier(n_estimators=num_trees, random_state=42)
    model.fit(features_train, target_train)
    
    train_predict = model.predict(features_train)
    train_accuracy.append(accuracy_score(target_train, train_predict))
    
    test_predict = model.predict(features_test)
    test_accuracy.append(accuracy_score(target_test, test_predict))

plt.figure(figsize=(10, 6))
plt.plot(num_trees_range, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(num_trees_range, test_accuracy, label='Testing Accuracy', marker='o')
plt.title('Random Forest Classifier - Accuracy Trends')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.show()