# The code is programmed by 
# Tang Longbin and Shi Jianrui
# for EES4408 Final Project.
import os
import svm
import knn
import torch
import logistic
import torchvision
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def mnist_data():
    '''
        return: train_data, test_data
    '''
    # Mnist digits dataset
    DATASET_DIR = './mnist/'
    DOWNLOAD_MNIST = False
    if not(os.path.exists(DATASET_DIR)) or not os.listdir(DATASET_DIR):
        # not mnist dir or mnist is empyt dir
        DOWNLOAD_MNIST = True
    # Load data
    train_data = torchvision.datasets.MNIST(root=DATASET_DIR, train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
    test_data = torchvision.datasets.MNIST(root=DATASET_DIR, train=False, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
    
    # # plot one random example
    # index = np.random.randint(0, len(train_data))
    # plt.imshow(train_input[index], cmap='gray')
    # plt.title('%i' % train_output[index])
    # plt.show()
    
    return train_data, test_data

def split_mnist_data(train_data, test_data):
    '''
        return: train_input, train_output, test_input, test_output
    '''
    # pick samples for training
    train_input = torch.unsqueeze(train_data.data, dim=1).type(torch.FloatTensor)/255.   # shape from (n, 28, 28) to (n, 1, 28, 28), value in range(0,1)
    train_input = train_input.view(len(train_data), -1)
    train_output = train_data.targets
    # pick samples for testing
    test_input = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255.   # shape from (n, 28, 28) to (n, 1, 28, 28), value in range(0,1)
    test_input = test_input.view(len(test_data), -1)
    test_output = test_data.targets
    
    print('Train Input:', train_input.size(),
          'Train Output:', train_output.size(),)
    print('Test Input:', test_input.size(),
          'Test Output:', test_output.size(),)
    print('Mnist Data Get!')
    
    return train_input, train_output, test_input, test_output

def main():
    print('-----------Load Mnist Data----------')
    train_data, test_data = mnist_data()
    train_input, train_output, test_input, test_output = split_mnist_data(train_data, test_data)
    print('----------------Done----------------')
    print('-------------Starting SVM-----------')
    svm.main(train_input, train_output, test_input, test_output)
    print('----------------Done----------------')
    print('-------------Starting KNN-----------')
    knn.main(train_input, train_output, test_input, test_output)
    print('----------------Done----------------')
    print('----Starting Logistic Regression----')
    logistic.main(train_data, test_data)
    print('----------------Done----------------')

if __name__ == '__main__':
    main()