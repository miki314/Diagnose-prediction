import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = read_csv("data/Training.csv")

# Creating X and Y

le = LabelEncoder()
le.fit(df["prognosis"])

x = df.copy()
x = x.drop("prognosis", axis=1)
y = le.transform(df["prognosis"])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)
#print(x.shape)
#print(y.shape)
#print(np.unique(y))

# Creating model
model = keras.models.Sequential([
    layers.Input(shape=(x.shape[1],)),
    layers.Normalization(),
    layers.Hashing(num_bins=3),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(np.unique(y).shape[0],activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=5)

# Testing the model
predictions = model.predict(x_test)
y_hat = []
y_confident = []

for i in range(0,predictions.shape[0],1):
    y_hat.append(np.argmax(predictions[i]))
    y_confident.append(np.max(predictions[i]))

# Print accuracy of model
print(accuracy_score(y_test, y_hat))

# Create confusion matrix
matrix = confusion_matrix(y_test, y_hat)
disp = ConfusionMatrixDisplay(matrix)
fig, ax = plt.subplots(figsize=(12,12))
disp.plot(ax=ax)
plt.show()
plt.close()

# Save model
model.save("models/3_64_relu_adam.keras")
