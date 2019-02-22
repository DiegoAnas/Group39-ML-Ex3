from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
import os


def flatten(l):
    return [item for sublist in l for item in sublist]


class FeatureExtractor:
    def __init__(self, imagePath: str, fileName:str):
        # the easiest way would to do the following:
        # imageOpenCV = cv2.imread(imagePath + fileName)

        # However, we have the same issue as before, and it is more difficult in OpenCV to convert to an RGB image
        # Thus we do this using PIL, and then convert to OpenCV ....
        #self.imagePIL = Image.open(imagePath + fileName)
        self.imagePIL = Image.open(fileName)
        self.imagePIL = self.imagePIL.convert('RGB')
        self.imageOpenCV = np.array(self.imagePIL)
        # Convert RGB to BGR
        self.imageOpenCV = self.imageOpenCV[:, :, ::-1].copy()

        # Now we split the image in the three channels, B / G / R
        self.chans = cv2.split(self.imageOpenCV)
        self.colors = ("b", "g", "r")

    def featureVectorPIL(self):
        # histogram over each RGB channel
        return self.imagePIL.histogram()

    def histogramFeaturesCV(self):
        featureVector1D = self.histFeature1D()
        featureVector2D = self.histFeature2D()
        featureVector3D = self.histFeature3D()
        return (featureVector1D, featureVector2D, featureVector3D)

    def histFeature1D(self) -> List[int]:
        featuresOpenCV_1D:List[int] = []
        bins_1D = 64
        for (chan, color) in zip(self.chans, self.colors):  # we compute the histogram over each channel
            histOpenCV = cv2.calcHist([chan], [0], None, [bins_1D], [0, 256])
            featuresOpenCV_1D.extend(histOpenCV)
        featureVectorOpenCV_1D = flatten(featuresOpenCV_1D)

        if (len(featureVectorOpenCV_1D) != bins_1D * 3):  # sanity check, in case we had a wrong number of channels...
            print("Unexpected length of feature vector: " + str(len(featureVectorOpenCV_1D)) + " in file: " + self.fileName)

        return featureVectorOpenCV_1D

    def histFeature2D(self):
        featuresOpenCV_2D: List[int] = [] #try
        bins2D = 16
        # look at all combinations of channels (R & B, R & G, B & G)
        featuresOpenCV_2D.extend(cv2.calcHist([self.chans[1], self.chans[0]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
        featuresOpenCV_2D.extend(cv2.calcHist([self.chans[1], self.chans[2]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
        featuresOpenCV_2D.extend(cv2.calcHist([self.chans[0], self.chans[2]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
        # and add that to our dataset
        featureVectorOpenCV_2D = flatten(featuresOpenCV_2D)

        return featureVectorOpenCV_2D

    def histFeature3D(self):
        # All three channels at the same time.
        # We further reduce our bin size, because otherwise, this would become very large...
        featuresOpenCV_3D = cv2.calcHist([self.imageOpenCV], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        featureVectorOpenCV_3D = featuresOpenCV_3D.flatten()

        return featureVectorOpenCV_3D

    def extractSiftFeatures(self) -> Tuple[List[int], List[int]]:
        gray = cv2.cvtColor(self.imageOpenCV, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        siftKeypoints, descriptors = sift.detectAndCompute(gray)

        return Tuple(siftKeypoints, descriptors)