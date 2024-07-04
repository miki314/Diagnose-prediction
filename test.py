import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

# Preparing data
df = read_csv("data/Training.csv")
le = LabelEncoder()
le.fit(df["prognosis"])
df = read_csv("data/custom_test.csv")

# Loading model
model = load_model("models/3_64_relu_adam.keras")
x = df.drop("prognosis",axis=1)
y = le.transform(df["prognosis"])

# Predicting data using neural network
predictions = model.predict(x)
y_results = [] # Second dimension is first and second choice

# Print the results
for i in range(0,predictions.shape[0],1):
    first = np.argmax(predictions[i])
    predictions[i][first] = 0
    second = np.argmax(predictions[i])
    y_results.append([first, second])

for i in y_results:
    prognosis = le.inverse_transform(i)
    print(prognosis)