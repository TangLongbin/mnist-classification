from sklearn import svm
from sklearn.metrics import confusion_matrix


def main(train_input, train_output, test_input, test_output):
    # Training SVM
    print('------Training and testing SVM------')
    clf = svm.SVC(C=5, gamma=0.05,max_iter=10)
    clf.fit(train_input, train_output)
    
    #Test on Training data
    train_result = clf.predict(train_input)
    precision = sum(train_result == train_output)/train_output.shape[0]
    print('Training precision: ', precision)
    
    #Test on test data
    test_result = clf.predict(test_input)
    precision = sum(test_result == test_output)/test_output.shape[0]
    print('Test precision: ', precision)
    
    #Show the confusion matrix
    matrix = confusion_matrix(test_output, test_result)
    print('Confusion matrix: ')
    print(matrix)