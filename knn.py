# The code is programmed by 
# Tang Longbin and Shi Jianrui
# for EES4408 Final Project.
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

def main(train_input, train_output, test_input, test_output):
    '''
        K-Nearest Neighbor Classification
    '''
    # split train data into train and validation
    n = int(len(train_input) * 5 / 6)
    trainData = np.array(train_input[:n])
    trainLabels = np.array(train_output[:n])
    valData = np.array(train_input[n:])
    valLabels = np.array(train_output[n:])
    testData = np.array(test_input)
    testLabels = np.array(test_output)

    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each value of k
    kVals = [3, 5, 7, 9, 15, 27, 49]
    accuracies = []
    print("Testing k in", kVals)
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