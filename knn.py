from __future__ import print_function

import os
import cv2
import time
import torch
import imutils
import torchvision
import numpy as np
import torch.nn as nn
from sklearn import datasets
from skimage import exposure
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def check_mnist():
    '''
        return: train_input, train_output, test_input, test_output
    '''
    # Mnist digits dataset
    DATASET_DIR = './mnist/'
    DOWNLOAD_MNIST = False
    if not(os.path.exists(DATASET_DIR)) or not os.listdir(DATASET_DIR):
        # not mnist dir or mnist is empyt dir
        DOWNLOAD_MNIST = True
    
    TRAIN_SAMPLES = 60000
    TEST_SAMPLES = 2000
    
    # pick TRAIN_SAMPLES samples for training
    train_data = torchvision.datasets.MNIST(root=DATASET_DIR, train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
    train_input = torch.unsqueeze(train_data.data, dim=1).type(torch.FloatTensor)[:TRAIN_SAMPLES]/255.   # shape from (TRAIN_SAMPLES, 28, 28) to (TRAIN_SAMPLES, 1, 28, 28), value in range(0,1)
    train_input = train_input.view(TRAIN_SAMPLES, -1)
    train_output = train_data.targets[:TRAIN_SAMPLES]

    # pick TEST_SAMPLES samples for testing
    test_data = torchvision.datasets.MNIST(root=DATASET_DIR, train=False, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
    test_input = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:TEST_SAMPLES]/255.   # shape from (TEST_SAMPLES, 28, 28) to (TEST_SAMPLES, 1, 28, 28), value in range(0,1)
    test_input = test_input.view(TEST_SAMPLES, -1)
    test_output = test_data.targets[:TEST_SAMPLES]
    
    # # plot one random example
    # index = np.random.randint(0, TRAIN_SAMPLES)
    # plt.imshow(train_input[index], cmap='gray')
    # plt.title('%i' % train_output[index])
    # plt.show()
    
    print('Train Input:', train_input.size(),
          'Train Output:', train_output.size(),)
    print('Test Input:', test_input.size(),
          'Test Output:', test_output.size(),)
    print('Mnist Data Get!')
    return train_input, train_output, test_input, test_output

def knn(train_input, train_output, test_input, test_output):
    '''
        K-Nearest Neighbor Classification
    '''
    
    trainData = np.array(train_input)
    trainLabels = np.array(train_output)
    testData = np.array(test_input)
    testLabels = np.array(test_output)
    valData = testData
    valLabels = testLabels
    
    
    # Checking sizes of each data split
    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))


    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each value of k
    kVals = range(1, 30, 2)
    accuracies = []
    print("Test k from", kVals)
    # loop over kVals
    for k in kVals:
        # train the classifier with the current value of `k`
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)

        # evaluate the model and print the accuracies list
        score = model.score(valData, valLabels)
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)
        localtime = time.asctime( time.localtime(time.time()) )
        print("Time:", localtime)

    # largest accuracy
    index = np.argmax(accuracies)
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[index], accuracies[index] * 100))
    
    # Now that I know the best value of k, re-train the classifier
    model = KNeighborsClassifier(n_neighbors=kVals[index])
    model.fit(trainData, trainLabels)

    # Predict labels for the test set
    predictions = model.predict(testData)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))


    # # some indices are classified correctly 100% of the time (precision = 1)
    # # high accuracy (98%)

    # # check predictions against images
    # # loop over a few random digits
    # image = testData
    # j = 0
    # for i in np.random.randint(0, high=len(testLabels), size=(24,)):
    #         # np.random.randint(low, high=None, size=None, dtype='l')
    #     prediction = model.predict(image)[i]
    #     image0 = image[i].reshape((8, 8)).astype("uint8")
    #     image0 = exposure.rescale_intensity(image0, out_range=(0, 255))
    #     # plt.subplot(4,6,j+1)
    #     # plt.title(str(prediction))
    #     # plt.imshow(image0,cmap='gray')
    #     # plt.axis('off')


    #         # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    #         # then resize it to 32 x 32 pixels for better visualization

    #         #image0 = imutils.resize(image[0], width=32, inter=cv2.INTER_CUBIC)

    #     j = j+1

    #     # show the prediction
    #     # print("I think that digit is: {}".format(prediction))
    #     # print('image0 is ',image0)
    #     # cv2.imshow("Image", image0)
    #     # cv2.waitKey(0) # press enter to view each one!
    # # plt.show()

def main():
    # check dataset
    train_input, train_output, test_input, test_output = check_mnist()
    # knn algorithm
    knn(train_input, train_output, test_input, test_output)

if __name__ == "__main__":
    main()