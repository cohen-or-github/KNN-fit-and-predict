import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p
        self.ids = (318403599, 206571135)
        self.train_points = []
        self.train_labels = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        self.train_points = X
        self.train_labels = y

    def find_label(self, labels_list):
        """
        This method finds the most common label in the list.
        If there's a tie, it's broken by given rules.
        :param labels_list: a list of labels and distances of each neighbour
        :return: the most common label
        """
        labels_dict = {}
        max_labels = []

        # initialize the dict
        for label in labels_list:
            labels_dict[label[0]] = (0, label[1])

        # update the dictionary for each label: add the counter of appearances and nearest point to test point
        for label in labels_list:
            labels_dict[label[0]] = (labels_dict[label[0]][0]+1, min(label[1], labels_dict[label[0]][1]))
        labels_sorted = sorted(labels_dict.items(), key=lambda x: x[1][0], reverse=True)

        # find all the labels with the max num of neighbours
        max_count = labels_sorted[0][1][0]
        for label in labels_sorted:
            if label[1][0] == max_count:
                max_labels.append(label)

        # return the label and break tie if needed
        return sorted(max_labels, key=lambda x: (x[1][1], x[0]))[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        test_labels = []
        for i, test_point in enumerate(X):
            # find each train point distance from the test point
            distances = []
            points_labels = []
            for j, train_point in enumerate(self.train_points):
                distances.append((j, np.linalg.norm(train_point - test_point, ord=self.p)))

            # add points labels to tuples
            for point in distances:
                points_labels.append((self.train_labels[point[0]], point[1]))

            # sort in order to determine in case of tie in distances and select k neighbours
            points_labels = sorted(points_labels, key=lambda x: (x[1], x[0]))
            k_neighbours = points_labels[:self.k]

            # find the predicted label of the test point and add the label to list
            test_labels.append(self.find_label(k_neighbours))
        return np.ndarray((X.shape[0], ), buffer=np.array(test_labels), dtype='uint8')


def main():

    print("*" * 20)
    print("Started HW1_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
