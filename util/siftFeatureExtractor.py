from typing import List
import cv2
import numpy as np
from PIL import Image

class siftFeatureExtractor:
    def __init__(self, imagePath: str, fileName:str):
        # the easiest way would to do the following:
        # imageOpenCV = cv2.imread(imagePath + fileName)
        self.imagePath = imagePath
        self.fileName = fileName

        img = cv2.imread(imagePath + fileName)
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def histFeature1D(self) -> List[int]:
        featuresOpenCV_1D:List[int] = []

        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(self.gray, None)



        # dataOpenCV_1D.append(featureVectorOpenCV_1D)  # now we append the feature vector to the dataset so far
        return featureVectorOpenCV_1D

