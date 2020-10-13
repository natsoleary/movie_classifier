# web scraping trial
import csv
from sys import stdin
import pandas


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def main():
    # ratings database
    filename = "RealRatings.csv"
    ratings = pandas.read_csv(filename)
    # print(ratings)
    rows = []
    count = 0
    lines = 0
    with open(filename, 'rt') as csv_file:

        # creating a csv reader object
        csvreader = csv.reader(csv_file, delimiter=',')
        # extracting each data row one by one
        try:
            i = 0
            while i < 5271:
                if not is_ascii(ratings['Title'][i]):
                    i += 1
                    continue
                title = ratings['Title'][i]
                rating = ratings['Rated'][i]
                i += 1
                title = title.replace(' ', '')
                title = title.lower()
                title = title.replace('.', '')
                title = title.replace(',', '')
                title = title.replace(':', '')
                title = title.replace("'", '')
                with open('all.txt') as file:
                    for line in file:
                        lines += 1
                        name = line.replace('.txt', '')
                        name = name.replace(' ',  '')

                        if name.strip().endswith('the'):
                            name = name[:-4]
                            name = 'the' + name
                        else:
                            name = name[:-1]
                        name = name.replace('.', '')
                        if title == name:
                            count += 1
                            print(line[:-1])
                            print(rating)

        except Exception as e:
            print(e)
        print(count)
        print(lines)


# -----------------------------------------------------------
if __name__ == '__main__':
    main()
