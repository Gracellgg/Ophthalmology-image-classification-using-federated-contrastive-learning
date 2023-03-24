import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# combine the model csv with the machine learning csv based on the data dir

def combine_csv(model_csv, ml_csv):
    model_csv = pd.read_csv(model_csv)
    ml_csv = pd.read_csv(ml_csv)
    combine = pd.merge(model_csv, ml_csv, on='data_dir')
    return combine




if __name__ == '__main__':

    combine_csv = combine_csv('GLCM.csv', 'feature_fusion.csv')

    # use svm to classify the combined csv file with 0.8 train and 0.2 test. The first column is the data dir, the second column is the label,
    # and the rest are the features
    train, test = train_test_split(combine_csv, test_size=0.5, random_state=42)
    train_x = train.iloc[:, 2:10]
    train_y = train.iloc[:, 1]
    test_x = test.iloc[:, 2:10]
    test_y = test.iloc[:, 1]
    #clf = svm.SVC(kernel='linear')
    clf = svm.SVC(kernel='rbf', C=10, gamma=0.1)
    clf.fit(train_x, train_y)
    print(clf.score(test_x, test_y))
    print(accuracy_score(test_y, clf.predict(test_x)))
    print(confusion_matrix(test_y, clf.predict(test_x)))






