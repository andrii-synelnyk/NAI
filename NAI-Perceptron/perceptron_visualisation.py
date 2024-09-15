import matplotlib.pyplot as plt
from perceptron_old import Perceptron, read_file, encode_categories  # Import from your perceptron_old.py


def main():
    # Load your dataset
    train_features, train_categories = read_file('perceptron.data')
    test_features, test_categories = read_file('perceptron.test.data')

    train_labels, category_to_int = encode_categories(train_categories)
    test_labels, _ = encode_categories(test_categories)

    learning_rate = 0.01  # Learning rate
    max_epochs = 1000  # Max number of epochs to train

    perceptron = Perceptron(len(train_features[0]), learning_rate)

    # Get accuracies over epochs
    accuracies = perceptron.train_and_evaluate(train_features, train_labels, test_features, test_labels, max_epochs)

    # Plotting
    plt.figure(figsize=(22, 6))
    plt.plot(range(1, max_epochs + 1), accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Perceptron Accuracy over Epochs')
    plt.show()


if __name__ == '__main__':
    main()