
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import io


PACKAGE_PATH = 'python_ocr/'
CHAR_WIDTH = 30


def __load_data():
    """ Load all OCR data sets """

    filename = '{}ocr_data.mat'.format(PACKAGE_PATH)
    mat_dict = io.loadmat(filename)
    return mat_dict


def load_train():
    """ Load the OCR training data """

    all_data = __load_data()
    train_data = all_data['train_data'].astype(np.float32)
    train_labels = all_data['train_labels']
    return train_data, train_labels[0, :]


def load_test():
    """ Load the OCR test data """

    all_data = __load_data()
    test_data = all_data['test_data'].astype(np.float32)
    test_labels = all_data['test_labels']
    return test_data, test_labels[0, :]


def load_challenge():
    """ Load the challenge test data """

    all_data = __load_data()
    challenge_data = all_data['challenge_data'].astype(np.float32)
    return challenge_data


def load_challenge_labels():
    """ Load the challenge test data """
    all_data = __load_data()
    challenge_labels = all_data['challenge_labels']
    return challenge_labels[0, :]


def display_character(char_data):
    """ Display a character from training/test data """
    letter_image = np.reshape(char_data,
                              (CHAR_WIDTH, CHAR_WIDTH), order='F')
    plt.matshow(letter_image, cmap=cm.Greys_r)


def classify(train, train_labels, test, features=None):
    """Nearest neighbour classification.

    train - data matrix storing training data, one sample per row
    train_label - a vector storing the training data labels
    test - data matrix storing the test data
    features - a vector if indices that select the feature to use
             if features=None then all features are used

    returns: labels - estimated test data labels
    """

    # Use all feature is no feature parameter has been supplied
    if features is None:
        features = np.arange(0, train.shape[1])

    # if only one test vector then make sure to cast into skinny matrix
    if test.ndim == 1:
        test = test[np.newaxis]

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]

    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test*test, axis=1))
    modtrain = np.sqrt(np.sum(train*train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argmax(dist, axis=1)
    labels = train_labels[nearest]

    return labels


def invert_data(data):
    return 768-data


def reflect_data(data):
    nrows, ncols = data.shape
    data = np.reshape(data, (30*nrows, 30))
    data = np.fliplr(data)
    data = np.reshape(data, (nrows, 900))
    return data


def evaluate(est_labels, true_labels):
    """Evaluate a classification result.

    est_labels - a vector storing the estimated labels
    true_labels - a vector storing the ground-truth labels

    returns: (score, confusions) - a percentage correct and a
                                  confusion matrix
    """

    n_labels = len(est_labels)

    score = (100.0 * sum(true_labels == est_labels))/len(est_labels)

    # Construct a confusion matrix
    nclasses = np.max(np.hstack((true_labels, est_labels)))
    confusions = np.zeros((nclasses, nclasses))
    for i in xrange(n_labels):
        confusions[true_labels[i]-1, est_labels[i]-1] += 1

    return score, confusions


def evaluate_standard_test(est_labels):

    __, test_labels = load_test()
    score, confusions = evaluate(est_labels, test_labels)
    return score, confusions
