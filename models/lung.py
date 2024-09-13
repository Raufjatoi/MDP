
import pandas as p
import numpy as n
import seaborn as s
import matplotlib.pyplot as m
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

data = p.read_csv("lung.csv")
#print(data.head())
#print(data.tail())
#data.tail()

data["LUNG_CANCER"].value_counts()

data["LUNG_CANCER"] = (data["LUNG_CANCER"] == "YES").astype(int)

for label in data.columns[:-1]:
  data[label] = (data[label]==2).astype(int)

#data.head()

#data['AGE'].value_counts()

#data['GENDER'].value_counts()

#data['SMOKING'].value_counts()

#data.head()
'''
for label in data.columns[:-1]:
  m.hist(data[data["LUNG_CANCER"]==1][label], color='red', label='have cancer', alpha=0.7, density=True)
  m.hist(data[data["LUNG_CANCER"]==0][label], color='blue', label='not cancer', alpha=0.7, density=True)
  m.title(label)
  m.ylabel("probability")
  m.xlabel(label)
  m.legend()
  m.show()
  '''

train, valid, test = n.split(data.sample(frac=1), [int(0.6*len(data)), int(0.8*len(data))])

def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = n.hstack((X, n.reshape(y, (-1, 1))))

  return data, X, y

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

#print(classification_report(y_test, y_pred))


from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
#print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)
#print(classification_report(y_test, y_pred))

from sklearn.svm import SVC

svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
#print(classification_report(y_test, y_pred))

import tensorflow as tf

import matplotlib.pyplot as plt

def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)

  plt.show()

# Convert X_train to a NumPy array and ensure it has the correct data type
X_train = n.array(X_train, dtype=n.float32)

# Convert y_train to a NumPy array and ensure it has the correct data type
y_train = n.array(y_train, dtype=n.float32)  # For binary classification

nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(32, activation='relu', input_shape=(15,)),
      #tf.keras.layers.Dropout(#),
      tf.keras.layers.Dense(32, activation='relu'),
      #tf.keras.layers.Dropout(#),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',
                  metrics=['accuracy'])
history = nn_model.fit(
    X_train, y_train, epochs=50, batch_size=32, validation_split=0.2
  )

#plot_history(history)

y_pred = nn_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

print(classification_report(y_test, y_pred))
nn_model.save('lung_cancer.h5')

print("model saved ")

