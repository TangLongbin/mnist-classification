import svm
import knn
import logistic

def main():
    print('----------Starting SVM----------')
    svm.main()
    print('----------Starting KNN----------')
    knn.main()
    print('----------Starting Logistic Regression----------')
    logistic.main()
    print('----------Done----------')

if __name__ == '__main__':
    main()