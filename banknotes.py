import random

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import csv

model = Perceptron()

# read data
with open("dataset/banknotes.csv") as f:
    reader = csv.reader(f)
    next(f)                             # skip the first line of the csv file

    data = []
    for row in reader:
        data.append({
            "evidence" : [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == 0 else "Counterfeit"
        })


# splitting data into training and test sets
holdout = int(0.5 * len(data))      # 50% of the data
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]


