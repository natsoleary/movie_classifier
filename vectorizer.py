import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split


def main():
    with open('corpus.pickle', 'rb') as handle:
        corpus = pickle.load(handle)
        with open('label.pickle', 'rb') as file:
            labels = pickle.load(file)
            print(labels)
            vectorizer = CountVectorizer()
            countMatrix = vectorizer.fit_transform(corpus)
            # print(countMatrix)
            tfidf = TfidfTransformer()
            features = tfidf.fit_transform(countMatrix, True)
            print(features)
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=.2)
            print(X_test)
            print(y_test)
            with open('X_test.pickle', 'wb') as handle:
                pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('y_test.pickle', 'wb') as handle:
                pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('X_train.pickle', 'wb') as handle:
                pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('y_train.pickle', 'wb') as handle:
                pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
