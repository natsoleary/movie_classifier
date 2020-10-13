import pickle
from sklearn.linear_model import LogisticRegression


def main():
    with open('X_test.pickle', 'rb') as handle:
        X_test = pickle.load(handle)
    with open('X_train.pickle', 'rb') as bhandle:
        X_train = pickle.load(bhandle)
    with open('y_test.pickle', 'rb') as chandle:
        y_test = pickle.load(chandle)
    with open('y_train.pickle', 'rb') as dhandle:
        y_train = pickle.load(dhandle)
    plot = LogisticRegression(
        solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    results = plot.predict(X_test)
    print(results)
    print(y_test)


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
