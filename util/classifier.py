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


class classifier:
    def __init__(self, datasets:List[List], labels:List[int], folds:int, KNN:bool=False, MLP:bool=False, RF:bool=False, SVM:bool=False):
        self.datasets = datasets
        self.labels = labels
        self.folds = folds
        self.classifiers = []
        self.classNames = []
        if KNN:
            self.classifiers.extend([
                neighbors.KNeighborsClassifier(weights='distance', n_neighbors=3),
                neighbors.KNeighborsClassifier(weights='distance', n_neighbors=5),
                neighbors.KNeighborsClassifier(weights='distance', n_neighbors=15),
                neighbors.KNeighborsClassifier(weights='uniform', n_neighbors=3),
                neighbors.KNeighborsClassifier(weights='uniform', n_neighbors=5),
                neighbors.KNeighborsClassifier(weights='uniform', n_neighbors=15),])
            self.classNames.extend(["Distance weighted 3-Knn",
                                    "Distance weighted 5-Knn",
                                    "Distance weighted 15-Knn",
                                    "Non-weighted 3-Knn",
                                    "Non-weighted 5-Knn",
                                    "Non-weighted 15-Knn",])
        if SVM:
            self.classifiers.extend([
                svm.SVC(kernel="linear", random_state=39),
                svm.SVC(kernel="poly", degree=3, random_state=39),
                svm.SVC(kernel="rbf", gamma="scale", random_state=39)])
            self.classNames.extend(["SVM with linear kernel",
                                    "SVM with 3rd-degree poly kernel",
                                    "SVM with RBF kernel",])
        if MLP:
            self.classifiers.extend([
                MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, random_state=39, tol=0.000000001),])
            self.classNames.extend(["Multilayer perceptron",])
        if RF:
            self.classifiers.extend([ensemble.RandomForestClassifier(random_state=39),])
            self.classNames.extend(["Random Forest",])

        if len(self.classifiers) == 0: # If no classifiers have been, perform 4 different ones
            self.classifiers = [
                    neighbors.KNeighborsClassifier(weights='distance', n_neighbors=3),
                    ensemble.RandomForestClassifier(random_state=39),
                    MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, random_state=39, tol=0.000000001),
                    svm.SVC(kernel="linear", random_state=39),]
            self.classNames = ["Weighted 3-Knn",
                               "Random Forest",
                               "Multilayer perceptron",
                               "Linear SVM"]

    def classify(self):
        for setIndex, dataset in enumerate(self.datasets):
            print(f"Set {setIndex}: \n")
            kFold = StratifiedKFold(n_splits=self.folds, shuffle=False, random_state=39)
            for clfIndex, clf in enumerate(self.classifiers):
                print(f"***** Classifier {self.classNames[clfIndex]}")
                bestF1 = 0
                fold = 1
                for train_index, test_index in kFold.split(dataset, self.labels):
                    print(f"Fold : {fold}")
                    trainingImages = [dataset[index] for index in train_index]
                    trainingLabels = [self.labels[index] for index in train_index]
                    testingImages = [dataset[index] for index in test_index]
                    testingLabels = [self.labels[index] for index in test_index]
                    startTimeSeconds = default_timer()
                    clf.fit(trainingImages, trainingLabels)
                    elapsedTimeSeconds = default_timer() - startTimeSeconds
                    print(f"Time to build model: {elapsedTimeSeconds}")
                    startTimeSeconds = default_timer()
                    prediction = clf.predict(testingImages)
                    elapsedTimeSeconds = default_timer() - startTimeSeconds
                    print(f"Time to make predictions: {elapsedTimeSeconds}")
                    if metrics.f1_score(testingLabels, prediction, average='micro') > bestF1:
                        bestF1 = metrics.f1_score(testingLabels, prediction, average='micro')
                        bestClassF1 = metrics.f1_score(testingLabels, prediction, average=None)
                        bestCM =  metrics.confusion_matrix(testingLabels, prediction)
                    fold += 1
                    print(f"Overall precision: {metrics.precision_score(testingLabels, prediction, average='micro')}")
                    print(f"Overall recall: {metrics.recall_score(testingLabels, prediction, average='micro')}")
                    print(f"Overall F1 score: {metrics.f1_score(testingLabels, prediction, average='micro')}")
                print(f"Results for best model:")
                print(f"Per class F1 score: \n {bestClassF1}")
                print(f"Confusion Matrix: \n {bestCM}\n ")