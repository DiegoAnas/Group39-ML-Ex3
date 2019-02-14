import glob, os, platform
from typing import List, Tuple
from sklearn import preprocessing
from util import demoPlot, FeatureExtractor, classifier
import cv2
import numpy as np
from PIL import Image
import datetime
from BoVW import *


class Ex3ML:
    @staticmethod
    def preprocess(self, imagePath:str) -> Tuple[List[str], List[int]]:

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


if __name__ == "__main__":

    imagePath = "FIDS30/"
    fileNames, target = Ex3ML.preprocess(imagePath)
    # Example plots moved to another file
    #demoPlot.main(imagePath + fileNames[1])

    dataPIL = []
    #dataOpenCV_1D = []
    #dataOpenCV_2D = []
    #dataOpenCV_3D = []
    dataSift = []
    for fileName in fileNames:
        extractor = FeatureExtractor.FeatureExtractor(imagePath, fileName)
        dataPIL.append(extractor.histogramFeaturesPIL())
        #dataOpenCV_3D.append(extractor.histFeature3D())
        #dataSift.append(extractor.extractSiftFeatures())

    print (".... done" + " (" + str(datetime.datetime.now()) + ")")

    trainingSets = [dataPIL]
    clf = classifier.classifier(trainingSets, target)
    clf.classify()
