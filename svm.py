from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def main(train_input, train_output, test_input, test_output):
    # torch tensor to numpy array
    train_input = train_input.numpy()
    train_output = train_output.numpy()
    test_input = test_input.numpy()
    test_output = test_output.numpy()
    
    # Training SVM
    print('------Training and testing SVM------')
    clf = svm.SVC(max_iter=3)
    clf.fit(train_input, train_output)
    
    #Test on Training data
    train_result = clf.predict(train_input)
    precision = sum(train_result == train_output)/len(train_output)
    
    print('Training precision: ', precision)
    
    #Test on test data
    test_result = clf.predict(test_input)
    precision = sum(test_result == test_output)/len(test_output)
    print('Test precision: ', precision)
    
    #Show the confusion matrix
    matrix = confusion_matrix(test_output, test_result)
    print('Confusion matrix: ')
    print(matrix)