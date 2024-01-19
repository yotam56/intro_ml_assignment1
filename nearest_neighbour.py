import numpy as np
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt

data = np.load('mnist_all.npz')


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m alongside its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two vectors.

    :param x1: First vector
    :param x2: Second vector
    :return: Euclidean distance
    """
    return np.linalg.norm(x1 - x2)


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classifier = {'k': k, 'x_train': x_train, 'y_train': y_train}
    return classifier


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learn knn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    predictions = []
    k = classifier['k']
    x_train = classifier['x_train']
    y_train = classifier['y_train']

    for test_vector in x_test:
        distances = [euclidean_distance(test_vector, x_train[i]) for i in range(len(x_train))]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])

    return np.array(predictions).reshape(-1, 1)


def simple_test():
    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def test_knn():
    x_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 1, 0, 1, 0])
    x_test = np.array([[1, 1], [4, 4], [6, 6]])
    expected_labels = np.array([0, 1, 0])
    classifier = learnknn(3, x_train, y_train)
    predicted_labels = predictknn(classifier, x_test)

    if np.array_equal(predicted_labels.flatten(), expected_labels):
        print("Test Passed: Predicted labels match the expected labels.")
    else:
        print("Test Failed: Predicted labels do not match the expected labels.")
        print("Predicted labels:", predicted_labels.flatten())
        print("Expected labels:", expected_labels)


def question_2a():
    # data = np.load("mnist_all.npz")
    sample_sizes = [5, 10, 20, 40, 60, 80, 100]
    # sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    average_errors = []
    error_min = []
    error_max = []

    for size in sample_sizes:
        errors = []
        for _ in range(10):
            x_train, y_train = gensmallm([data["train2"], data["train3"], data["train5"], data["train6"]],
                                         [2, 3, 5, 6], size)
            x_test, y_test = gensmallm([data["test2"], data["test3"], data["test5"], data["test6"]],
                                       [2, 3, 5, 6], size)

            classifier = learnknn(1, x_train, y_train)
            y_test_pred = predictknn(classifier, x_test)

            error = np.mean(y_test != list(np.concatenate(y_test_pred).flat))
            errors.append(error)

        average_errors.append(np.mean(errors))
        error_min.append(np.min(errors))
        error_max.append(np.max(errors))

    plt.scatter(sample_sizes, error_max, color="green")
    plt.scatter(sample_sizes, error_min, color="red")
    plt.scatter(sample_sizes, average_errors, color="blue")
    x = sample_sizes
    y = average_errors
    up = [item1 - item2 for item1, item2 in zip(error_max, average_errors)]
    down = [item1 - item2 for item1, item2 in zip(average_errors, error_min)]
    yerr = [down, up]
    plt.xlabel('Training Sample Size')
    plt.ylabel('Average Test Error')
    plt.title('KNN Test Error vs Training Sample Size')
    plt.errorbar(x, y, yerr=yerr, capsize=3, fmt="r--o", ecolor="black")
    plt.show()


def question_2e():
    # data = np.load("mnist_all.npz")
    average_errors = []
    error_min = []
    error_max = []
    k_nums = [k for k in range(1, 12)]
    for k in k_nums:
        errors = []
        for _ in range(10):
            x_train, y_train = gensmallm([data["train2"], data["train3"], data["train5"], data["train6"]],
                                         [2, 3, 5, 6], 200)
            x_test, y_test = gensmallm([data["test2"], data["test3"], data["test5"], data["test6"]],
                                       [2, 3, 5, 6], 200)

            classifier = learnknn(k, x_train, y_train)
            y_test_pred = predictknn(classifier, x_test)

            error = np.mean(y_test != list(np.concatenate(y_test_pred).flat))
            errors.append(error)

        average_errors.append(np.mean(errors))
        error_min.append(np.min(errors))
        error_max.append(np.max(errors))

    plt.scatter(k_nums, error_max, color="green")
    plt.scatter(k_nums, error_min, color="red")
    plt.scatter(k_nums, average_errors, color="blue")
    x = k_nums
    y = average_errors
    up = [item1 - item2 for item1, item2 in zip(error_max, average_errors)]
    down = [item1 - item2 for item1, item2 in zip(average_errors, error_min)]
    yerr = [down, up]
    plt.xlabel('K size')
    plt.ylabel('Average Test Error')
    plt.title('KNN Test Error vs K Size')
    plt.errorbar(x, y, yerr=yerr, capsize=3, fmt="r--o", ecolor="black")
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    # test_knn()
    # question_2a()
    question_2e()
