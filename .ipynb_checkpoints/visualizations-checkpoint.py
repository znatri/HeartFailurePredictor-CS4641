# Author: Felipe Mairhofer
# Date: 11/12/2023
# Description: This file contains the functions that are used to process the Kaggle Dataset
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('data/heart.csv')
print(df.head())
print(df.dtypes)

print(df.isnull().sum())

features = df.iloc[:, :-1]
target = df.iloc[:, -1]

features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=0.15)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

model = RandomForestClassifier(n_estimators=100)

model.fit(features_train, target_train)

target_predict = model.predict(features_test)

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
