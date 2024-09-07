import pandas as p
import numpy as n 
import matplotlib.pyplot as pl
import seaborn as sns
data = p.read_csv('stroke.csv')
#data.head()
#data.info()
#data.describe()
'''
pl.figure(figsize=(8, 5))
pl.title('Count of Stroke Cases')
sns.countplot(x='stroke', data=data)
pl.show()
pl.figure(figsize=(8, 5))
pl.title('Gender Distribution')
sns.countplot(x='gender', data=data)
pl.show()
pl.figure(figsize=(8, 5))
pl.title('Marital Status Distribution')
sns.countplot(x='ever_married', data=data)
pl.show()
pl.figure(figsize=(8, 5))
pl.title('Work Type Distribution')
sns.countplot(x='work_type', data=data)
pl.show()
pl.figure(figsize=(8, 5))
pl.title('Age Distribution')
sns.histplot(x='age', data=data, bins=30)
pl.show()
pl.figure(figsize=(8, 5))
pl.title('Average Glucose Level Distribution')
sns.histplot(x='avg_glucose_level', data=data, bins=30)
pl.show()
pl.figure(figsize=(8, 5))
pl.title('BMI Distribution')
sns.histplot(x='bmi', data=data, bins=30)
pl.show()

'''

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score

# Prepare the data
X = data.drop('stroke', axis=1)
y = data['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(silent=True)
}


from sklearn.impute import SimpleImputer


# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

results = {}
for model_name, model in models.items():
    model.fit(X_train_imputed, y_train)
    y_pred = model.predict(X_test_imputed)
    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'ROC AUC': roc_auc,
        'Confusion Matrix': conf_matrix,
        'Classification Report': class_report,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall
    }
'''
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"ROC AUC: {metrics['ROC AUC']:.4f}")
    print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")
    print(f"Classification Report:\n{metrics['Classification Report']}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}\n")

    ''' 

'''
results_df = p.DataFrame(results).T
results_df[['Accuracy', 'ROC AUC', 'F1 Score', 'Precision', 'Recall']].plot(kind='bar', figsize=(12, 6))
pl.title('Model Performance Comparison')
pl.ylabel('Score')
pl.xticks(rotation=45)
pl.grid(axis='y')
pl.show()
'''
# saving lightgbm , catboost , xgboost models
import joblib

joblib.dump(models['catboost'], 'catboost_model.pkl')
joblib.dump(models['lightgbm'], 'lightgbm_model.pkl')
joblib.dump(models['xgboost'], 'xgboost_model.pkl')

print("models saved")
