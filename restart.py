import glob
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def main():
    path = '/u/noleary/indepwork/imsdb_raw_nov_2015/Feasible/*.txt'
    files = glob.glob(path)
    # iterate over the list getting each file
    corpus = []
    labels = []
    titles = []
    for fle in files:
        # open the file and then call .read() to get the text
        # query the label file to append labels to the labels
        # handle the error of not finding the file
        substrings = fle.split('/')
        title = substrings[-1]
        with open(fle, 'r') as f:
            text = f.read()
            with open('trimmed.txt') as file:
                for line in file:
                    line = line[:-1]
                    if str(line) == str(title):
                        label = next(file)[:-1]
                        label = label.lstrip()
                        labels.append(label)
                        corpus.append(text)
                        titles.append(str(title))
                        break
    if len(labels) != len(corpus):
        print('not same length')
        # print(features)
    print(labels)
    i = 0
    for l in labels:
        if l != 'R':
            if l != 'PG-13':
                if l != 'PG':
                    print('WRONG!!' + l)
                    print(i)
                    labels[i] = 'PG-13'
        i += 1
    X_train, X_test, y_train, y_test = train_test_split(
        corpus, labels, test_size=.2)
    vectorizer = CountVectorizer()
    countMatrix = vectorizer.fit_transform(corpus)
    countArray = countMatrix.toarray()
    tfidf = TfidfTransformer()
    features = tfidf.fit_transform(countMatrix, True)
    featureArray = features.toarray()
    testTitles = []
    trainTitles = []
    testVectors = []
    trainVectors = []
    testLabels = []
    trainLabels = []

    for x in X_test:
        i = 0
        while i < len(corpus):
            if corpus[i] == x:
                testTitles.append(titles[i])
                testVectors.append(featureArray[i])
                testLabels.append(labels[i])
            i += 1
    for x in X_train:
        i = 0
        while i < len(corpus):
            if corpus[i] == x:
                trainTitles.append(titles[i])
                trainVectors.append(featureArray[i])
                trainLabels.append(labels[i])
            i += 1
    print(testTitles)
    # print(testVectors)
    print(y_test)
    print("vs")
    print(testLabels)
    allVectors = testVectors + trainVectors

    plot = LogisticRegression(
        solver='lbfgs', multi_class='multinomial').fit(trainVectors, y_train)
    results = plot.predict(testVectors)
    print(results)
    cscore = plot.score(testVectors, y_test)
    print(cscore)
    print(len(testVectors))
    print(len(y_test))
    print(len(testLabels))
    cscores = plot.decision_function(testVectors)
    print(cscores)

    y_scores = MultiLabelBinarizer().fit_transform(results)
    y_true = MultiLabelBinarizer().fit_transform(y_test)
    roc = roc_auc_score(y_true, y_scores)
    print(roc)
    print("what the fuck")
    numbers = []
    for r in labels:
        if r == 'PG':
            numbers.append(0)
        if r == 'PG-13':
            numbers.append(1)
        if r == 'R':
            numbers.append(2)
    if len(numbers) != len(labels):
        print('uh oh')
    print(numbers)

    # Binarize the output
    y = label_binarize(np.array(numbers), classes=[0, 1, 2])
    n_classes = y.shape[1]

    X = np.array(allVectors)
    n_samples, n_features = X.shape

    print("got here 1 ")
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=0)
    print("got here 2")
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(
        svm.SVC(kernel='linear', probability=True))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    print("got here 3")
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print("got here 4")

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('firstgraph.svg')


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
