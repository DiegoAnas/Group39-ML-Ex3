import glob, os, platform
from typing import List, Tuple
from sklearn import preprocessing
from util import demoPlot, FeatureExtractor, classifier
from util.classifiers import BagWords
import cv2
import numpy as np
from PIL import Image
import datetime
from BoVW import *
from timeit import default_timer


class Ex3ML:
    @staticmethod
    def preprocess(imagePath:str) -> Tuple[List[str], List[int]]:

        ## Create the create the ground truth (label assignment, target, ...)
        imagePath = "FIDS30/"  # path to our image folder
        # Find all images in that folder
        os.chdir(imagePath)
        fileNames = glob.glob("*/*.jpg")
        numberOfFiles = len(fileNames)
        targetLabels = []

        print("Found " + str(numberOfFiles) + " files\n")
        # The first step - create the ground truth (label assignment, target, ...)
        # For that, iterate over the files, and obtain the class label for each file
        # Basically, the class name is in the full path name, so we simply use that

        for fileName in fileNames:
            if platform.system() == "Windows":
                pathSepIndex = fileName.index("\\")
            else:
                pathSepIndex = fileName.index("/")
            targetLabels.append(fileName[:pathSepIndex])

        # sk-learn can only handle labels in numeric format - we have them as strings though...
        # Thus we use the LabelEncoder, which does a mapping to Integer numbers

        le = preprocessing.LabelEncoder()
        le.fit(targetLabels)  # this basically finds all unique class names, and assigns them to the numbers
        print("Found the following classes: " + str(list(le.classes_)))

        # now we transform our labels to integers
        target = le.transform(targetLabels);
        print("Transformed labels (first elements: " + str(target[0:150]))

        # If we want to find again the label for an integer value, we can do something like this:
        # print list(le.inverse_transform([0, 18, 1]))
        print("... done label encoding")
        return fileNames, target

    @staticmethod
    def experimentPIL(fileNames: List[str], labels: List[int]):
        print("Extracting features using PIL.\n")
        startTimeSeconds = default_timer()
        dataPIL = []
        for fileName in fileNames:
            extractor = FeatureExtractor.FeatureExtractor(imagePath, fileName)
            dataPIL.append(extractor.featureVectorPIL())

        elapsedTimeSeconds = default_timer() - startTimeSeconds
        print(f"Time to extract histogram features using PIL: {elapsedTimeSeconds}")
        print(f"Names {np.shape(fileNames)} ")
        print(f"data {np.shape(dataPIL)} ")
        print(f"labels {np.shape(labels)} ")
        clf = classifier.classifier(dataPIL, labels)
        clf.classify()

    @staticmethod
    def experimentCVHist(fileNames: List[str], labels: List[int]):
        print("Extracting features using OpenCV.\n")
        startTimeSeconds = default_timer()
        dataOpenCV_1D = []
        dataOpenCV_2D = []
        dataOpenCV_3D = []
        for fileName in fileNames:
            extractor = FeatureExtractor.FeatureExtractor(imagePath, fileName)
            dataOpenCV_1D.append(extractor.histFeature1D())
            dataOpenCV_2D.append(extractor.histFeature2D())
            dataOpenCV_3D.append(extractor.histFeature3D())

        elapsedTimeSeconds = default_timer() - startTimeSeconds
        print(f"Time to extract histogram features using OpenCV: {elapsedTimeSeconds}")
        trainingSets = [dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D]
        clf = classifier.classifier(trainingSets, labels)
        clf.classify()

    @staticmethod
    def experimentVisualBagOfWords(fileNames: List[str], labels: List[int]):
        bag = BagWords.BagOfWords(fileNames, labels)
        bag.run()

if __name__ == "__main__":

    imagePath = "FIDS30/"
    fileNames, labels = Ex3ML.preprocess(imagePath)

    # Example plots moved to another file
    #demoPlot.main(imagePath + fileNames[1])

    #TODO parsing commands

    Ex3ML.experimentPIL(fileNames, labels)
    Ex3ML.experimentCVHist(fileNames, labels)
    Ex3ML.experimentVisualBagOfWords(fileNames, labels)
