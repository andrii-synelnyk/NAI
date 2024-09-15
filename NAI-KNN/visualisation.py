import matplotlib.pyplot as plt
import time
from main import read_file, classify, accuracy


def plot_k_vs_accuracy(train_features, train_categories, test_features, test_categories, max_k):
    k_values = range(1, max_k + 1)
    accuracies = []

    start_time = time.time()  # Start time measurement

    for k in k_values:
        predictions = classify(k, train_features, train_categories, test_features)
        acc = accuracy(predictions, test_categories)
        accuracies.append(acc)
        print(f"k={k}, Accuracy={acc}")

    end_time = time.time()  # End time measurement
    total_time = end_time - start_time  # Calculate total elapsed time
    print(f"Total time to get all predictions: {total_time:.2f} seconds")

    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('k-NN: Number of Neighbors vs. Accuracy (Iris Dataset)')
    plt.show()


def main(train_filename, test_filename, max_k):
    train_features, train_categories = read_file(train_filename)
    test_features, test_categories = read_file(test_filename)
    plot_k_vs_accuracy(train_features, train_categories, test_features, test_categories, max_k)


if __name__ == '__main__':
    # train_filename = input("Enter the filename of the train data: ")
    # test_filename = input("Enter the filename of the test data: ")
    # max_k = int(input("Enter the maximum number of neighbours to be checked: "))

    max_k = 100
    train_filename = "iris.data"
    test_filename = "iris.test.data"

    main(train_filename, test_filename, max_k)