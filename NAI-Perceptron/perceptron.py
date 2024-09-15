import random
import csv


class Perceptron:
    def __init__(self, num_inputs, learning_rate):
        # Initialize weights and bias to random values between 0 and 1
        self.weights = [random.uniform(0, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(0, 1)
        self.learning_rate = learning_rate

    # The predict function uses a linear combination of weights and inputs, adding bias, and applies a step function
    def predict(self, inputs):
        summation = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return 1 if summation >= 0 else 0  # step function used for activation

    # Train the perceptron over a specified number of epochs; updates weights and bias based on the error
    def train(self, training_data, labels, epochs):
        for epoch in range(epochs):
            # Error is summed over all training examples for each epoch
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                # Adjust weights and bias using Delta Rule
                self.weights = [w + self.learning_rate * error * i for w, i in zip(self.weights, inputs)]
                self.bias += self.learning_rate * error

    # Train the model for each epoch and evaluate on test data; track accuracy for each epoch
    def train_and_evaluate(self, training_data, train_labels, test_data, test_labels, epochs):
        accuracies = []
        for epoch in range(epochs):
            self.train(training_data, train_labels, 1)  # Train for one epoch
            predictions = [self.predict(feature) for feature in test_data]
            epoch_accuracy = accuracy(predictions, test_labels)
            accuracies.append(epoch_accuracy)
        return accuracies


# Reads a CSV file and separates it into features and labels
def read_file(filename):
    with open(filename, 'r') as file:
        csvreader = csv.reader(file)
        features, categories = [], []
        for row in csvreader:
            row_features = []
            for i in range(len(row) - 1):  # Iterate over all elements except the last one
                row_features.append(float(row[i]))  # Convert each feature to float and add to the list

            category = row[-1]  # The last element is the category

            features.append(row_features)
            categories.append(category)
    return features, categories


# Encodes categorical labels into integer values for model training
def encode_categories(categories):
    unique_categories = list(set(categories))
    category_to_int = {cat: i for i, cat in enumerate(unique_categories)}  # dictionary comprehension {'Iris-versicolor': 0, 'Iris-virginica': 1}
    return [category_to_int[cat] for cat in categories], category_to_int  # return list of categories converted to their corresponding int values and dictionary for future decoding


# Decodes the numerical predictions back into categorical labels (used only for user input vectors)
def decode_prediction(prediction, category_to_int):
    int_to_category = {v: k for k, v in category_to_int.items()}  # creating the inverse of the category_to_int
    return int_to_category[prediction]


# Calculates accuracy as the proportion of correct predictions
def accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))  # using generator expression to count the number of when p == l is true
    return correct / len(predictions)


def main():
    # User input for learning rate and number of epochs, and data file loading
    train_filename = "perceptron.data"
    test_filename = "perceptron.test.data"
    learning_rate = float(input("Enter the learning rate: "))
    epochs = int(input("Enter the number of epochs: "))

    train_features, train_categories = read_file(train_filename)
    test_features, test_categories = read_file(test_filename)

    # Encode the categorical labels for training and testing
    train_labels, category_to_int = encode_categories(train_categories)
    test_labels, _ = encode_categories(test_categories)

    # Initialize and train the perceptron model
    perceptron = Perceptron(len(train_features[0]), learning_rate)
    perceptron.train(train_features, train_labels, epochs)

    # Evaluate the model on the test set and print accuracy
    predictions = [perceptron.predict(feature) for feature in test_features]
    # decoded_predictions = [decode_prediction(p, category_to_int) for p in predictions]
    # decoded_test_labels = [decode_prediction(l, category_to_int) for l in test_labels]

    print("Test Accuracy:", accuracy(predictions, test_labels))

    # Interactive loop for manual data entry and classification
    while True:
        user_input = input("\nEnter a new vector to classify (or 'exit' to stop): ")
        if user_input.lower() == 'exit':
            break
        vector = [float(x) for x in user_input.split(',')]
        prediction = perceptron.predict(vector)
        class_label = decode_prediction(prediction, category_to_int)
        print("Predicted Class:", class_label)


if __name__ == '__main__':
    main()
