# Author: Nicholas Arribasplata
# Date: 12/03/2023
# Description: This file contains the linear regression model for our dataset

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart.csv')
print(df.head())
print(df.dtypes)
print(df.isnull().sum())

numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

for feature in numerical_features:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
    df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])

features = df.iloc[:, :-1]
target = df.iloc[:, -1]

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.15)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

logistic_model = LogisticRegression()

logistic_model.fit(features_train, target_train)

# Make predictions on the test set
target_predict = logistic_model.predict(features_test)

# Evaluate the model performance
accuracy = accuracy_score(target_test, target_predict)
confusion_mat = confusion_matrix(target_test, target_predict)
class_report = classification_report(target_test, target_predict)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", confusion_mat)
print("Classification Report:\n", class_report)

coefficients = logistic_model.coef_[0]
feature_names = features.columns

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Coefficients
plt.figure(figsize=(10, 6))
plt.barh(range(len(coefficients)), coefficients)
plt.yticks(range(len(coefficients)), feature_names)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Logistic Regression - Coefficients')
plt.show()

# Extracting Precision, Recall, and F1-score
report_data = classification_report(target_test, target_predict, output_dict=True)
precision = report_data['weighted avg']['precision']
recall = report_data['weighted avg']['recall']
f1_score = report_data['weighted avg']['f1-score']

# Creating lists for the metrics and scores
metrics = ['Accuracy', 'Recall', 'F1-score', 'Precision']
scores = [accuracy, recall, f1_score, precision]

plt.figure(figsize=(10, 6))

# Plotting the grouped bar plot
sns.barplot(x=metrics, y=scores, palette='muted')
plt.ylim(0, 1)
plt.title('Model Metrics')
plt.ylabel('Score')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(range(len(target_test)), target_test, color='blue', label='True Values', alpha=0.5)
plt.scatter(range(len(target_predict)), target_predict, color='red', label='Predictions', alpha=0.5)
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('True Values vs Predictions')
plt.legend()
plt.show()

train_accuracy = []
test_accuracy = []
model = LogisticRegression()

for num_trees in range(1, 101):
    model.fit(features_train, target_train)
    
    train_predict = model.predict(features_train)
    train_accuracy.append(accuracy_score(target_train, train_predict))
    
    test_predict = model.predict(features_test)
    test_accuracy.append(accuracy_score(target_test, test_predict))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_accuracy, label='Training Accuracy', marker='o')
plt.plot(range(1, 101), test_accuracy, label='Testing Accuracy', marker='o')
plt.title('Logistic Regression - Accuracy Trends')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
