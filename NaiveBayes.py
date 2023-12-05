import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('data/heart.csv')
print(df.head())
print(df.dtypes)

print(df.isnull().sum())

features = df.iloc[:, :-1]
target = df.iloc[:, -1]

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.15)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

model = GaussianNB()  # Use Gaussian Naive Bayes

model.fit(features_train, target_train)

target_predict = model.predict(features_test)

accuracy = accuracy_score(target_test, target_predict)
confusion_mat = confusion_matrix(target_test, target_predict)
class_report = classification_report(target_test, target_predict)

print("Accuracy:", accuracy)
print("Confusion matrix:\n", confusion_mat)
print("Classification Report:\n", class_report)

# Visualizations
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Accuracy Visualization
plt.figure(figsize=(6, 4))
sns.barplot(x=['Accuracy'], y=[accuracy], palette='viridis')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.show()

# Extracting Precision, Recall, and F1-score
report_data = classification_report(target_test, target_predict, output_dict=True)
precision = report_data['weighted avg']['precision']
recall = report_data['weighted avg']['recall']
f1_score = report_data['weighted avg']['f1-score']

# Precision, Recall, and F1-score Visualization
labels = ['Precision', 'Recall', 'F1-score']
scores = [precision, recall, f1_score]

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=scores, palette='muted')
plt.ylim(0, 1)
plt.title('Precision, Recall, and F1-score')
plt.show()
