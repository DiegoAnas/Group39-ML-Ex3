from typing import List
import cv2
import numpy as np
from PIL import Image

def flatten(l):
    return [item for sublist in l for item in sublist]

class siftFeatureExtractor:
    def __init__(self, imagePath: str, fileName:str):
        # the easiest way would to do the following:
        # imageOpenCV = cv2.imread(imagePath + fileName)
        self.imagePath = imagePath
        self.fileName = fileName

        img = cv2.imread(imagePath + fileName)
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def extractSiftFeatures(self) -> List[int]:
        featuresOpenCV_1D:List[int] = []

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(self.gray, None)

        featuresOpenCV_1D.extend(des)
        featureVectorOpenCV_1D = flatten(featuresOpenCV_1D)
        return featureVectorOpenCV_1D

