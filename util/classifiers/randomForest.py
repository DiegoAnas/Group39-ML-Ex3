from sklearn import ensemble
from typing import List

class randomForest:
    def __init__(self):
        self.classifier = ensemble.RandomForestClassifier()

    def train(self, labels:List[int], data):
        pass

