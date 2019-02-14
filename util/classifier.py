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
from sklearn.neural_network import MLPClassifier
from sklearn import svm

class classifier:
    def __init__(self, trainingSets:List[List], labels:List[int]):
        self.trainingSets = trainingSets
        self.labels = labels
        self.classifiers = [
            neighbors.KNeighborsClassifier(weights='distance', n_neighbors=17),
            # naive_bayes.GaussianNB(),
            ensemble.RandomForestClassifier(random_state=39),
            MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, random_state=39,tol=0.000000001)
            # svm.SVC(),
            # svm.LinearSVC(),
        ]
        self.names: List[str]=["Knn","RandomForest","MLP"]

    def classify(self):
        counter=0
        for index, train in enumerate(self.trainingSets):
            i_train, i_test, l_train, l_test = \
                train_test_split(train, self.labels, test_size=0.25, random_state=39, stratify=self.labels)
            counter=0
            print("******************************************")
            for classifier in self.classifiers:
                print(self.names[counter])
                counter = counter+1
                classifier.fit(i_train, l_train)
                prediction = classifier.predict(i_test)
                #print(f""" Confusion matrix: \n {metrics.confusion_matrix(l_test, prediction)} \n
                #{metrics.f1_score(l_test, prediction, average=None)}
                #""")
                print(classifier.score(i_test, l_test))
                # apply model to test date
                # measure accuracy
                # confusion matrix!!!!1