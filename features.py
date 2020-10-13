import glob
import pickle

from sklearn.feature_extraction.text import CountVectorizer


def main():
    path = '/u/noleary/indepwork/imsdb_raw_nov_2015/Feasible/*.txt'
    files = glob.glob(path)
    # iterate over the list getting each file
    corpus = []
    labels = []
    data = []
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

                        break

    # check len(corpus) == len(labels)
    if len(labels) != len(corpus):
        print("Something went wrong")
    print("corpus length", len(corpus))
    print("labels length", len(labels))
    print(labels)
    with open('corpus.pickle', 'wb') as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # numbers = []
    # for label in labels:
    #     if 'PG-13' in label:
    #         numbers.append(1)
    #     elif 'PG' in label:
    #         numbers.append(0)
    #     elif label == 'R':
    #         numbers.append(2)
    # print(numbers)
    # print(len(numbers))
    # print(len(labels))
    # with open('nlabel.pickle', 'wb') as handle:
    #     pickle.dump(numbers, handle, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
