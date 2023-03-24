import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

# combine the model csv with the machine learning csv based on the data dir

def combine_csv(model_csv, ml_csv):
    model_csv = pd.read_csv(model_csv)
    ml_csv = pd.read_csv(ml_csv)
    combine = pd.merge(model_csv, ml_csv, on='data_dir')
    return combine




if __name__ == '__main__':

    test = combine_csv('GLCM.csv', 'simCLR_feature_fusion.csv')
    train = combine_csv('GLCM.csv', 'simCLR_feature_fusion_train.csv')

    # use svm to classify the combined csv file with 0.8 train and 0.2 test. The first column is the data dir, the second column is the label,
    # and the rest are the features
    #train, test = train_test_split(combine_csv, test_size=0.1, random_state=42)

    train_x = train.iloc[:, 24:]
    train_y = train.iloc[:, 1]
    test_x = test.iloc[:, 24:]
    test_y = test.iloc[:, 1]
    clf = svm.SVC(kernel='linear')
    #clf = svm.SVC(kernel='rbf', C=10, gamma=0.1)
    #clf = LogisticRegression()

    clf.fit(train_x, train_y)
    print(clf.score(test_x, test_y))
    print(accuracy_score(test_y, clf.predict(test_x)))
    # print precision and recall

    print(classification_report(test_y, clf.predict(test_x)))

    print(confusion_matrix(test_y, clf.predict(test_x)))

    # plot the confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_y, clf.predict(test_x))
    np.set_printoptions(precision=2)

    plt.imshow(cnf_matrix, cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks([0,1,2])
    plt.yticks([0,1,2])
    for i in range(len(cnf_matrix)):
        for j in range(len(cnf_matrix[i])):
            # make the text white if the background is dark
            if cnf_matrix[i][j] > 0.5 * cnf_matrix.max():
                plt.text(i, j, cnf_matrix[i][j], fontsize=35, horizontalalignment='center', verticalalignment='center', color='white')
            else:

                plt.text(i, j, cnf_matrix[i][j], fontsize=35, horizontalalignment='center', verticalalignment='center')

    # save plot
    plt.savefig('confusion_matrix.png', dpi=300)

    plt.show()












