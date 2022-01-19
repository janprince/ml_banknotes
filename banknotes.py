import random

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import csv

model = SVC()

# read data
with open("dataset/banknotes.csv") as f:
    reader = csv.reader(f)
    next(f)                             # skip the first line of the csv file

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })


# splitting data into training and test sets
holdout = int(0.5 * len(data))      # 50% of the data
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]


# Train model on training set
X_training = [row["evidence"] for row in training]
Y_training = [row["label"] for row in training]
model.fit(X_training, Y_training)


# Make predictions on the testing set
X_testing = [row["evidence"] for row in testing]
Y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)


# Compute how well we performed
correct = 0
incorrect = 0
total = 0

for actual, predicted in zip(Y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1


# Print results
print(f"Results for model: {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * (correct/total):.2f}")
