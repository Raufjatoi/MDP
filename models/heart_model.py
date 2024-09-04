import pandas as p
import numpy as n
import matplotlib.pyplot as pl
import seaborn as s

data = p.read_csv('heart.csv')
#data.head()

#data.isnull().sum()

#data.describe()

#data.info()

#pl.figure(figsize=(12, 8))
#s.histplot(data['age'], bins=20, kde=True)
#pl.title('Age Distribution')
#pl.show()

#pl.figure(figsize=(12, 8))
#s.histplot(data['chol'], bins=20, kde=True)
#pl.title('Cholesterol Distribution')
#pl.show()

#pl.figure(figsize=(14, 10))
#s.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
#pl.title('Correlation Matrix')
#pl.show()

#s.pairplot(data, hue='target', palette='Set1')
#pl.show()

#pl.figure(figsize=(12, 8))
#s.countplot(data=data, x='sex', hue='target')
#pl.title('Heart Disease Frequency by Sex')
#pl.show()

#pl.figure(figsize=(12, 8))
#s.countplot(data=data, x='cp', hue='target')
#pl.title('Heart Disease Frequency by Chest Pain Type')
#pl.show()


data['sex'] = data['sex'].map({0: 'female', 1: 'male'})
data = p.get_dummies(data, drop_first=True)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

scaler = StandardScaler()
data_scaled = p.DataFrame(scaler.fit_transform(data.drop('target', axis=1)), columns=data.columns[:-1])
data_scaled['target'] = data['target']

X = data_scaled.drop('target', axis=1)
y = data_scaled['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X


lr = LogisticRegression()
#lr.fit(X_train, y_train)
#y_pred_lr = lr.predict(X_test)
#print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_lr))

"""## Random forest classifier"""

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_rfc))

"""## support vector machines svc"""

#svc = SVC()
#svc.fit(X_train, y_train)
#y_pred_svc = svc.predict(X_test)
##print('SVC Accuracy:', accuracy_score(y_test, y_pred_svc))

"""## Gradient Boosting Classifier"""

#gbc = GradientBoostingClassifier()
#gbc.fit(X_train, y_train)
#y_pred_gbc = gbc.predict(X_test)
#print('Gradient Boosting Accuracy:', accuracy_score(y_test, y_pred_gbc))

"""## Knn ( k nearest neighbor )"""

#knn = KNeighborsClassifier()
#knn.fit(X_train, y_train)
#y_pred_knn = knn.predict(X_test)
#print('KNN Accuracy:', accuracy_score(y_test, y_pred_knn))

"""## deeplearning model"""

#model = Sequential()
#model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
#model.add(Dropout(0.5))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

#model_loss, model_accuracy = model.evaluate(X_test, y_test)
#print('Neural Network Accuracy:', model_accuracy)

"""## results"""
'''
cm = confusion_matrix(y_test, y_pred_lr)  # For Logistic Regression
pl.figure(figsize=(10, 6))
s.heatmap(cm, annot=True, fmt='d', cmap='Blues')
pl.title('Confusion Matrix for Logistic Regression')
pl.show()

models = ['Logistic Regression', 'Random Forest', 'SVC', 'Gradient Boosting', 'KNN', 'Neural Network']
accuracies = [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rfc),
              accuracy_score(y_test, y_pred_svc), accuracy_score(y_test, y_pred_gbc),
              accuracy_score(y_test, y_pred_knn), model_accuracy]

pl.figure(figsize=(12, 8))
s.barplot(x=models, y=accuracies, palette='viridis')
pl.title('Model Accuracy Comparison')
pl.show()
'''
"""## Saving the models"""

''' # Save Logistic Regression model
joblib.dump(lr, 'logistic_regression_model.pkl')


# Save SVC model
joblib.dump(svc, 'svc_model.pkl')

# Save Gradient Boosting model
joblib.dump(gbc, 'gradient_boosting_model.pkl')

# Save KNN model
joblib.dump(knn, 'knn_model.pkl')

# Save Neural Network model
model.save('neural_network_model.h5')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

'''
joblib.dump(rfc, 'random_forest_model.pkl')

print("All models have been saved successfully. just random forest lol cause it was the best of all in this data  ")