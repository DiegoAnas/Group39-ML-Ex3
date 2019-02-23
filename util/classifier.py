from typing import List

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import neighbors
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from timeit import default_timer
import numpy as np
import textwrap

class classifier:
    def __init__(self, datasets:List[List], labels:List[int]):
        self.datasets = datasets
        self.labels = labels
        self.classifiers = [
            neighbors.KNeighborsClassifier(weights='distance', n_neighbors=17),
            # naive_bayes.GaussianNB(),
            ensemble.RandomForestClassifier(random_state=39),
            MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, random_state=39,tol=0.000000001),
            svm.SVC(),
            # svm.LinearSVC(),
        ]
        self.names: List[str]=["Knn","RandomForest","MLP", "SVM"]

    def classify(self):
        #TODO replace CrossValidation with K-Fold CV
        for setIndex, dataset in enumerate(self.datasets):
            # trainingImages, testingImages, trainingLabels, testingLabels = \
            #     train_test_split(dataset, self.labels, test_size=0.3, random_state=39, stratify=self.labels)
            kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=39)
            for train_index, test_index in kfold.split(dataset, self.labels):
                print(type(dataset))
                trainingImages = [dataset[index] for index in train_index]
                trainingLabels = [self.labels[index] for index in train_index]
                testingImages = [dataset[index] for index in test_index]
                testingLabels = [self.labels[index] for index in test_index]
                print(f"Set {setIndex}: \n")
                for clfIndex, classifier in  enumerate(self.classifiers):
                    print(f"Classifier {self.names[clfIndex]}\n")
                    startTimeSeconds = default_timer()
                    classifier.fit(trainingImages, trainingLabels)
                    elapsedTimeSeconds = default_timer() - startTimeSeconds
                    print(f"Time to build model: {elapsedTimeSeconds}")
                    startTimeSeconds = default_timer()
                    prediction = classifier.predict(testingImages)
                    elapsedTimeSeconds = default_timer() - startTimeSeconds
                    print(f"Time to make predictions: {elapsedTimeSeconds}")
                    print(f"Overall F1 score: {metrics.f1_score(testingLabels, prediction, average='micro')} \n")
                    print(f"Per class F1 score: \n {metrics.f1_score(testingLabels, prediction, average=None)}\n")
                    # print(f"Confusion Matrix: \n {metrics.confusion_matrix(testingLabels, prediction)}\n ")