#All Necessary Modules Import
import numpy as np
from sklearn.model_selection import train_test_split                                                                    #Test train Split
from sklearn import datasets                                                                                            #Using Sklearn Modules To Create own Datasets
import matplotlib.pyplot as plt

from naiveclass import NaiveBayes                                                                                       #Using opps concept to inherit the class

def accuracyScore(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=32000, n_features=15, n_classes=2, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

model_nb = NaiveBayes()
model_nb.fit(X_train, y_train)
predictions = model_nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracyScore(y_test, predictions)*100)