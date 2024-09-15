import csv
from collections import defaultdict
import math


# Function to read the dataset
def read_file(filename):
    with open(filename, 'r') as file:
        csvreader = csv.reader(file)
        data = []
        for row in csvreader:
            data.append(row)
    return data


# Function to separate the features and the labels
def separate_features_labels(data):
    features = [row[1:] for row in data]
    labels = [row[0] for row in data]
    return features, labels


# Function to calculate prior probabilities
def calculate_prior_probabilities(labels):
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    total_count = len(labels)
    priors = {label: count / total_count for label, count in label_counts.items()}
    return priors


# Function to calculate likelihood probabilities with Laplace smoothing
def calculate_likelihood_probabilities(features, labels):
    likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    label_counts = defaultdict(int)
    feature_values = defaultdict(set)

    for i in range(len(features)):
        label = labels[i]
        label_counts[label] += 1
        for j in range(len(features[i])):
            feature = features[i][j]
            likelihoods[label][j][feature] += 1
            feature_values[j].add(feature)

    for label in likelihoods:
        for j in likelihoods[label]:
            total_count = label_counts[label]
            num_values = len(feature_values[j])
            for feature in feature_values[j]:
                likelihoods[label][j][feature] = (likelihoods[label][j][feature] + 1) / (total_count + num_values)

    return likelihoods


# Function to classify a new instance using the computed probabilities
def classify_instance(instance, priors, likelihoods):
    best_label = None
    max_prob = -math.inf
    for label in priors:
        log_prob = math.log(priors[label])
        for j in range(len(instance)):
            feature = instance[j]
            if feature in likelihoods[label][j]:
                log_prob += math.log(likelihoods[label][j][feature])
            else:
                log_prob += math.log(1 / (sum(likelihoods[label][j].values()) + len(likelihoods[label][j])))
        if log_prob > max_prob:
            max_prob = log_prob
            best_label = label
    return best_label


# Function to evaluate the model
def evaluate(predictions, labels):
    tp = fp = tn = fn = 0
    for i in range(len(predictions)):
        if predictions[i] == 'p' and labels[i] == 'p':
            tp += 1
        elif predictions[i] == 'p' and labels[i] == 'e':
            fp += 1
        elif predictions[i] == 'e' and labels[i] == 'e':
            tn += 1
        elif predictions[i] == 'e' and labels[i] == 'p':
            fn += 1

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return accuracy, precision, recall, f_measure


# Main function
def main(train_filename, test_filename):
    # Read and preprocess the data
    train_data = read_file(train_filename)
    test_data = read_file(test_filename)

    train_features, train_labels = separate_features_labels(train_data)
    test_features, test_labels = separate_features_labels(test_data)

    # Calculate probabilities
    priors = calculate_prior_probabilities(train_labels)
    likelihoods = calculate_likelihood_probabilities(train_features, train_labels)

    # Classify the test set
    predictions = []
    for instance in test_features:
        prediction = classify_instance(instance, priors, likelihoods)
        predictions.append(prediction)

    # Evaluate the classifier
    accuracy, precision, recall, f_measure = evaluate(predictions, test_labels)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-measure: {f_measure}")


# Run the main function
if __name__ == "__main__":
    # train_filename = input("Enter the filename of the train data: ")
    # test_filename = input("Enter the filename of the test data: ")

    train_filename = "agaricus-lepiota.data"
    test_filename = "agaricus-lepiota.test.data"

    main(train_filename, test_filename)
