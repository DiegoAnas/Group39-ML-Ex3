from typing import List
from BoVW.helpers import BOVHelpers
from ..FeatureExtractor import FeatureExtractor
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

class BagOfWords:
    def __init__(self, fileNames: List[str], target: List[int]):
        self.fileNames = fileNames
        self.labels = target
        self.descriptors = None
        self.no_clusters = 100      # No idea what number
        self.bov_helper = BOVHelpers(self.no_clusters)
        pass

    def work(self):
        self.getDescriptors()
        train_descriptors, test_descriptors, train_labels, test_labels = \
            train_test_split(self.descriptors, self.labels, test_size=0.25, random_state=39, stratify=self.labels)
        self.trainModel(train_descriptors, len(train_descriptors), train_labels)
        self.testModel(test_descriptors, test_labels)
        pass

    def getDescriptors(self):
        descriptors = []
        sift = cv2.xfeatures2d.SIFT_create()
        for file in self.fileNames:
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            descriptors.append(des)
        self.descriptors = descriptors
        pass

    def trainModel(self, descriptors, imageCount, labels):
        self.bov_helper.formatND(self.descriptors)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images = imageCount, descriptor_list=self.descriptors)
        self.bov_helper.standardize()
        self.bov_helper.train(labels)
        pass

    def predict(self, descriptor):
        # generate vocab for test image
        vocab = np.array([[0 for i in range(self.no_clusters)]])
        # locate nearest clusters for each of
        # the visual word (feature) present in the image
        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(descriptor)
        # print(test_ret)
        for each in test_ret:
            vocab[0][each] += 1
        #print(vocab)

        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)

        # predict the class of the image
        predictedLabel = self.bov_helper.clf.predict(vocab)
        # print("Image belongs to class : ", self.name_dict[str(int(lb[0]))]

        return predictedLabel

    def testModel(self, descriptors, imageCount, labels):
        #Not implemented
        pass