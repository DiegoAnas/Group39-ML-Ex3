import glob, os, platform
from typing import List, Tuple
from sklearn import preprocessing
from util import demoPlot, FeatureExtractor, classifier, ConvolutionalNN
from util.classifiers import BagWords
import cv2
import numpy as np
from PIL import Image
from BoVW import *
from timeit import default_timer
import pickle   # Used for persistence
import argparse


class Ex3ML:
    @staticmethod
    def preprocess(imagePath:str) -> Tuple[List[str], List[int]]:
        ## Create the create the ground truth (label assignment, target, ...)
        imagePath = "FIDS30/"  # path to our image folder
        # Find all images in that folder
        try:
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
        except Exception as ex:
            print(f"An exception ocurred {ex}")
            exit(-1)

    @staticmethod
    def buildPILHist(fileNames: List[str]):
        print("Extracting features using PIL.\n")
        startTimeSeconds = default_timer()
        dataPIL = []
        for index, fileName in enumerate(fileNames):
            imagePIL = Image.open(fileName)
            # Not all images in our dataset are in RGB color scheme (e.g. indexed colours)
            # We need to make sure that they are RGB , otherwise we can't expect to have exactly three RGB channels..
            imagePIL = imagePIL.convert('RGB')
            featureVector = imagePIL.histogram()
            if (len(
                    featureVector) != 768):  # just a sanity check; with the transformation to RGB, this should never happen
                print(f"Unexpected length of feature vector: {str(len(featureVector))} in file: {fileName}")
            dataPIL.append(featureVector)

        elapsedTimeSeconds = default_timer() - startTimeSeconds
        print(f"Time to extract histogram features using PIL: {elapsedTimeSeconds}")
        try:
            with open('PILHist.data', 'wb') as fp:
                pickle.dump(dataPIL, fp)
        except Exception as ex:
            print(f"Exception occurrred writing files: \n {ex}")
        return dataPIL

    @staticmethod
    def loadPILHist():
        print(f"Loading PIL histogram data from disk")
        try:
            with open('PILHist.data', 'rb') as fp:
                dataPIL = pickle.load(fp)
        except Exception as ex:
            print(f"Exception occurrred: \n {ex}")
            return None, None, None
        return dataPIL

    @staticmethod
    def experimentPIL(fileNames: List[str], labels:List[int], folds:int,
                      KNN:bool=False, MLP:bool=False, RF:bool=False, SVM:bool=False):
        dataPIL = Ex3ML.loadCVHist()
        if dataPIL is None:
            dataPIL = Ex3ML.buildPILHist(fileNames)
        if dataPIL is None:
            print("Could not load or build dataset")
        clf = classifier.classifier([dataPIL], labels, folds=folds, KNN=KNN, MLP=MLP, RF=RF, SVM=SVM)
        clf.classify()

    @staticmethod
    def buildCVHist(fileNames: List[str]):
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
        print(f"Writing feature vectors to disk")
        try:
            with open('dataOpenCV_1D', 'wb') as fp:
                pickle.dump(dataOpenCV_1D, fp)
            with open('dataOpenCV_2D', 'wb') as fp:
                    pickle.dump(dataOpenCV_2D, fp)
            with open('dataOpenCV_3D', 'wb') as fp:
                pickle.dump(dataOpenCV_3D, fp)
        except Exception as ex:
            print(f"Exception occurrred writing files: \n {ex}")
        return dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D

    @staticmethod
    def loadCVHist():
        print(f"Loading feature vectors from disk")
        try:
            with open('dataOpenCV_1D', 'rb') as fp:
                dataOpenCV_1D = pickle.load(fp)
            with open('dataOpenCV_2D', 'rb') as fp:
                dataOpenCV_2D = pickle.load(fp)
            with open('dataOpenCV_3D', 'rb') as fp:
                dataOpenCV_3D = pickle.load(fp)
        except Exception as ex:
            print(f"Exception occurrred: \n {ex}")
            return None, None, None
        return dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D

    @staticmethod
    def experimentCVHist(fileNames, labels:List[int], folds:int,
                         KNN:bool=False, MLP:bool=False, RF:bool=False, SVM:bool=False):
        dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D = Ex3ML.loadCVHist()
        if dataOpenCV_1D is None:
            dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D = Ex3ML.buildCVHist(fileNames)
        if dataOpenCV_1D is None:
            print("Could not load or build dataset")
        else:
            datasets = [dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D]
            clf = classifier.classifier(datasets, labels, folds = folds, KNN=KNN, MLP=MLP, RF=RF, SVM=SVM)
            clf.classify()

    @staticmethod
    def experimentVisualBagOfWords(fileNames: List[str], labels: List[int]):
        bag = BagWords.BagOfWords(fileNames, labels)
        bag.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "ML3App",
        description="Image classfication experiment",
        epilog = "Application implemented for the 3rd assignment of ML course")
    parser.add_argument('imagePath', action='store',
                        help='Path to where the images are stored')
    parser.add_argument('--openCVHist', action='store_true',
        help='Perform classification using histogram features from OpenCV')
    parser.add_argument('--PILHist', action='store_true',
        help='Perform classification using histogram features from PILLOW')
    parser.add_argument('--BOVW', action='store_true',
        help='Perform classification using the Bag of Visual Words method')
    parser.add_argument('--CNN', action='store_true',
                        help='Perform classification using Convolutional Neural Network')
    parser.add_argument('--RF', action='store_true',
                        help='Test different RandomForest models')
    parser.add_argument('--SVM', action='store_true',
                        help='Test different SVM models')
    parser.add_argument('--KNN', action='store_true',
                        help='Test different KNN models')
    parser.add_argument('--MLP', action='store_true',
                        help='Test different MLP models')
    parser.add_argument('-f', "--folds", type=int, default=2, action='store', choices=range(2, 10),
                        help='Number of folds for CrossValidation (default 2).')
    parser.add_argument('--gray', action='store_true',
                        help='Convert images to grayscale (only applies to CNN experiment)')
    args = parser.parse_args()

    imagePath = args.imagePath
    #imagePath = "FIDS30/"
    fileNames, labels = Ex3ML.preprocess(imagePath)
    # TODO parsing commands

    # Example plots moved to another file
    #demoPlot.main(imagePath + fileNames[1])

    if args.PILHist:
        Ex3ML.experimentPIL(fileNames, labels, folds=args.folds, KNN=args.KNN, MLP=args.MLP, RF=args.RF, SVM=args.SVM)
    if args.openCVHist:
        Ex3ML.experimentCVHist(fileNames, labels, folds=args.folds, KNN=args.KNN, MLP=args.MLP, RF=args.RF, SVM=args.SVM)
    if args.BOVW:
        Ex3ML.experimentVisualBagOfWords(fileNames, labels)
    if args.CNN:
        ConvolutionalNN.experimentCNN(fileNames, labels, colour=not(args.gray))
