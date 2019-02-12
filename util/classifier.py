from typing import List

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import neighbors
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn import svm

class classifier:
    def __init__(self, trainingSets:List[List]):
        self.trainingSets = trainingSets
        self.classifiers = [
            neighbors.KNeighborsClassifier(),
            naive_bayes.GaussianNB(),
            tree.DecisionTreeClassifier(),
            ensemble.RandomForestClassifier(),
            svm.SVC(),
            svm.LinearSVC(),
        ]

    def classify(self):
        for index, train in enumerate(self.trainingSets):
            for classifier in self.classifiers:
                pass
        # do the classification here ....