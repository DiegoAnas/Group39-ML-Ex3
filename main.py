import glob, os, platform
from typing import List, Tuple
from sklearn import preprocessing
from util import demoPlot, featureExtractor, classifier, siftFeatureExtractor
import cv2
import numpy as np
from PIL import Image
import datetime


## Create the create the ground truth (label assignment, target, ...)
imagePath="FIDS30/"                 # path to our image folder
# Find all images in that folder
os.chdir(imagePath)
fileNames = glob.glob("*/*.jpg")
numberOfFiles=len(fileNames)
targetLabels=[]

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
le.fit(targetLabels) # this basically finds all unique class names, and assigns them to the numbers
print ("Found the following classes: " + str(list(le.classes_)))

# now we transform our labels to integers
target = le.transform(targetLabels);
print("Transformed labels (first elements: " + str(target[0:150]))

# If we want to find again the label for an integer value, we can do something like this:
# print list(le.inverse_transform([0, 18, 1]))
print("... done label encoding")

# Example plots moved to another file
demoPlot.main(imagePath + fileNames[1])

##--------------------------------------------
#Feature extraction
# so NOW we actually extract features from our images

print("Extracting features using PIL/PILLOW" + " (" + str(datetime.datetime.now()) + ")")

# The simplest approach is via the PIL/PILLOW package; here we get a histogram over each RGB channel
# Note: this doesn't really represent colours, as a colour is made up of the combination of the three channels!
data:List[int] = []     #try
for index, fileName in enumerate(fileNames):
    imagePIL = Image.open(imagePath + fileName)
    # Not all images in our dataset are in RGB color scheme (e.g. indexed colours)
    # We need to make sure that they are RGB , otherwise we can't expect to have exactly three RGB channels..
    imagePIL = imagePIL.convert('RGB')
    featureVector = imagePIL.histogram()

    if (len(featureVector) != 768):  # just a sanity check; with the transformation to RGB, this should never happen
        print(f"Unexpected length of feature vector: { str(len(featureVector)) } in file: { fileName}")

    data.append(featureVector)

# Next, we extract a few more features using OpenCV

print("Extracting features using OpenCV" + " (" + str(datetime.datetime.now()) + ")")
#dataOpenCV_1D = []
#dataOpenCV_2D = []
#dataOpenCV_3D = []
#
#for fileName in fileNames:
#    extractor = featureExtractor.featureExtractor(imagePath, fileName)
#    #dataOpenCV_3D.append(extractor.histFeature3D())
#    features:Tuple = extractor.histogramFeatures()
#    dataOpenCV_1D.append(features[0])
#    dataOpenCV_2D.append(features[1])
#    dataOpenCV_3D.append(features[2])
#
print (".... done" + " (" + str(datetime.datetime.now()) + ")")

print("Extracting features using SIFT" + " (" + str(datetime.datetime.now()) + ")")
dataSift=[]
for fileName in fileNames:
    extractor=siftFeatureExtractor.siftFeatureExtractor(imagePath, fileName)
    dataSift.append(extractor.extractSiftFeatures())

print (".... done" + " (" + str(datetime.datetime.now()) + ")")

trainingSets = [data,dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D]
#trainingSets = [data]
clf = classifier.classifier(trainingSets, target)
clf.classify()
