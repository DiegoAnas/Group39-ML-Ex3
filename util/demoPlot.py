# Example plots
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
from PIL import Image

def main(demoImage):
    # For Notebook
    #%matplotlib inline

    # For OpenCV (need Version 2.4+) for Python 2.7
    os.chdir("..")
    print ("Showing demo feature extraction on image " + demoImage)

    # load the image & plot it
    imagePIL = Image.open(demoImage)
    imgplot = plt.imshow(imagePIL)
    plt.title(demoImage)

    # now we compute a colour histogram using the histogram function in pillow
    # This gives us one histogram with 768 values, which is 3 x 256 values for each colour
    # For each colour channel, each value repesent the count how many pixels have that colour intensity
    featureVector = imagePIL.histogram()

    # We plot this histogram
    plt.figure()
    plt.plot(featureVector[:256], 'r')
    plt.plot(featureVector[257:512], 'g')
    plt.plot(featureVector[513:], 'b')
    plt.xlim([0, 256])
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.title("Colour Histogram, using PIL")

    # An alternative is to use open CV
    imageOpenCV = cv2.imread(demoImage)

    # OpenCV is a bit weird, because it changes the channel order, it stores them as BGR, instead of RGB
    # So if we want to display the image, we have to invert it
    # Plots image 2nd time?
    # plt.figure()
    # plt.imshow(cv2.cvtColor(imageOpenCV, cv2.COLOR_BGR2RGB))

    chans = cv2.split(imageOpenCV)  # split the image in the different channels (RGB, but in open CV, it is BGR, actually..)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Colour Histogram, using OpenCV")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    featuresOpenCV = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and add it to the resulting histograms array (of arrays)
        # We can specifiy here in the 4th argument how many bins we want - 256 means the same as in the previous histogram
        histOpenCV = cv2.calcHist([chan], [0], None, [256], [0, 256])
        featuresOpenCV.extend(histOpenCV)

        # plot the histogram of the current colour
        plt.plot(histOpenCV, color=color)
        plt.xlim([0, 256])

    # Now we have a 2D-array - 256 values for each of 3 colour channels.
    # To input this into our machine learning, we need to "flatten" the features into one larger 1D array
    # the size of this will be 3 x 256 = 768 values
    featureVectorOpenCV = np.array(featuresOpenCV).flatten()

    # show all the plots
    plt.show()

if __name__ == "__main__":
    main()