import timeit
import os
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore', category=Warning, append=True)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss

__author__ = 'Saurabh Rewaskar'

"""
Author: Saurabh Rewaskar (rewaskarsaurabh@outlook.com)
The program implements a Machine Learning algorithm for email categorization/ classification 
into bundles.
"""

NEWLINE = '\n'


def build_data_frame(path):
    """
    Creates DataFrame from data given at path.
    :param path: The path from which to read the data.
    :return: data_frame: The DataFrame created from the data.
    """
    rows = []
    index = []
    for file_name, text, classification, frequency in read_emails_function(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame


def read_emails_function(path):
    """
    This function reads emails and cats files from all the directories. 
    :param path: path to the main directory
    :return: None
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            temp_var = os.path.splitext(name)
            # email body file
            if temp_var[1] == '.txt':
                entire_path_of_file = os.path.join(root, name)

                if os.path.isfile(entire_path_of_file):

                    list_of_lines = list()
                    header = False
                    file_open = open(entire_path_of_file)

                    for line in file_open:
                        if header:
                            list_of_lines.append(line)
                        elif line == '\n':
                            header = True

                    file_open.close()
                    content = NEWLINE.join(list_of_lines)
                    # corresponding category file
                    with open(os.path.join(root, temp_var[0] + '.cats')) as cat_file:
                        classification = []
                        frequency = []
                        for line in cat_file:
                            line = line.strip().split(',')
                            classification.append(",".join(line[0:2]))
                            frequency.append(line[-1])
                    yield entire_path_of_file, content, classification, frequency


def classify(pipeline, classes, data):
    """
    Classifies the data using the pipeline given into the classes given.
    :param pipeline: The pipeline of text_extraction and classifier used to classify the data.
    :param classes: The classes on which the classification is done.
    :param data: The data on which the classification is done
    :return: None
    """
    k_fold = KFold(n_splits=5)
    start = timeit.default_timer()
    foldno = 1
    for train_indices, test_indices in k_fold.split(data):
        # Training set
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values

        # convert to sparse matrix form
        mlb = MultiLabelBinarizer(classes)
        train_y_mlb = mlb.fit_transform(train_y)

        # Testing set
        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values

        # convert to sparse matrix form
        mlb = MultiLabelBinarizer(classes)
        test_y_mlb = mlb.fit_transform(test_y)

        # train
        pipeline.fit(train_text, train_y_mlb)
        # predict
        predictions = pipeline.predict(test_text)
        print("Hamming Loss for Fold", foldno, ":", hamming_loss(test_y_mlb, predictions))
        foldno += 1

    stop = timeit.default_timer()
    print('Time taken:', stop - start)
    print("\n")


def main():
    """
    Main function of the program. This function makes calls to other functions.
    :return: None
    """

    files = [
        'data/enron_with_categories/1',
        'data/enron_with_categories/2',
        'data/enron_with_categories/3',
        'data/enron_with_categories/4',
        'data/enron_with_categories/5',
        'data/enron_with_categories/6',
        'data/enron_with_categories/7',
        'data/enron_with_categories/8',
    ]

    data = DataFrame({'text': [], 'class': []})

    # create data frame
    for path in files:
        data = data.append(build_data_frame(path))

    # The classes as mentioned in the categories.txt file,
    # we ignore the third parameter in the cats file
    classes = ("1,1", "1,2", "1,3", "1,4", "1,5", "1,6", "1,7", "1,8",
               "2,1", "2,2", "2,3", "2,4", "2,5", "2,6", "2,7", "2,8", "2,9", "2,10", "2,11", "2,12", "2,13",
               "3,1", "3,2", "3,3", "3,4", "3,5", "3,6", "3,7", "3,8", "3,9", "3,10", "3,11", "3,12", "3,13",
               "4,1", "4,2", "4,3", "4,4", "4,5", "4,6", "4,7", "4,8", "4,9", "4,10",
               "4,11", "4,12", "4,13", "4,14", "4,15", "4,16", "4,17", "4,18", "4,19")

    # Create pipelines for each combination of text_extraction and classifier

    pipeline = Pipeline([
        ('text_extraction', CountVectorizer(ngram_range=(2, 2))),
        ('classifier', OneVsRestClassifier(MultinomialNB()))
    ])
    print("text_extraction: ", "CountVectorizer", "classifier:", "MultinomialNB")
    classify(pipeline, classes, data)

    pipeline = Pipeline([
        ('text_extraction', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', OneVsRestClassifier(LinearSVC()))
    ])
    print("text_extraction: ", "CountVectorizer", "classifier:", "LinearSVC")
    classify(pipeline, classes, data)

    pipeline = Pipeline([
        ('text_extraction', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', OneVsRestClassifier(KNeighborsClassifier()))
    ])
    print("text_extraction: ", "CountVectorizer", "classifier:", "KNeighborsClassifier")
    classify(pipeline, classes, data)

    pipeline = Pipeline([
        ('text_extraction', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', OneVsRestClassifier(MultinomialNB()))
    ])
    print("text_extraction: ", "TfidfVectorizer", "classifier:", "MultinomialNB")
    classify(pipeline, classes, data)

    pipeline = Pipeline([
        ('text_extraction', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', OneVsRestClassifier(LinearSVC()))
    ])
    print("text_extraction: ", "TfidfVectorizer", "classifier:", "LinearSVC")
    classify(pipeline, classes, data)

    pipeline = Pipeline([
        ('text_extraction', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', OneVsRestClassifier(KNeighborsClassifier()))
    ])
    print("text_extraction: ", "TfidfVectorizer", "classifier:", "KNeighborsClassifier")
    classify(pipeline, classes, data)


if __name__ == '__main__':
    main()
