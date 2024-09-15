import csv


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


def euclidean_distance(instance1, instance2):
    # Calculate the Euclidean distance between two instances
    distance = 0
    for i in range(len(instance1)):
        distance += (instance1[i] - instance2[i]) ** 2
    return distance ** 0.5  # square root


def get_neighbors(train_features, test_instance, k):
    # Find k nearest neighbors of test_instance among the train_features
    distances = []
    for i in range(len(train_features)):
        dist = euclidean_distance(test_instance, train_features[i])
        distances.append((train_features[i], dist, i))
    # bubble_sort(distances)
    quick_sort(distances)
    neighbors = []
    # Get first k neighbors with the smallest distances
    for i in range(k):
        if i < len(distances):
            neighbors.append(distances[i])
    return neighbors


# def bubble_sort(arr, descending=False):
#     n = len(arr)
#     for i in range(n):
#         for j in range(0, n - i - 1):
#             if descending:
#                 if arr[j][1] < arr[j + 1][1]:  # Sort in descending order
#                     arr[j], arr[j + 1] = arr[j + 1], arr[j]
#             else:
#                 if arr[j][1] > arr[j + 1][1]:  # Sort in ascending order
#                     arr[j], arr[j + 1] = arr[j + 1], arr[j]


def quick_sort(arr, descending=False):
    def partition(low, high):
        pivot = arr[high][1]  # second element of the last array is chosen as pivot
        i = low - 1  # boundary pointer 'i' is initialized to one position before the start of the segment
        for j in range(low, high):
            if (descending and arr[j][1] > pivot) or (not descending and arr[j][1] < pivot):
                i = i + 1  # move boundary to indicate new fitting element is found
                arr[i], arr[j] = arr[j], arr[i]  # swap elements
        arr[i + 1], arr[high] = arr[high], arr[i + 1]  # place pivot in the correct position (right after boundary 'i')
        return i + 1

    def quick_sort_recursive(low, high):
        if low < high:
            pi = partition(low, high)
            quick_sort_recursive(low, pi - 1)
            quick_sort_recursive(pi + 1, high)

    quick_sort_recursive(0, len(arr) - 1)


def classify(number_of_k, train_features, train_categories, test_features):
    predictions = []
    for test_instance in test_features:
        # Finding neighbours for a test_instance
        neighbors = get_neighbors(train_features, test_instance, number_of_k)
        # Deciding for a category for a test_instance
        votes = {}
        for neighbor in neighbors:
            response = train_categories[neighbor[2]]  # Get the category of the neighbor
            votes[response] = votes.get(response, 0) + 1  # Add vote for the category
        votes_list = list(votes.items())
        # bubble_sort(votes_list, descending=True)
        quick_sort(votes_list, descending=True)
        highest_vote_category = votes_list[0][0]
        predictions.append(highest_vote_category)
    return predictions


def accuracy(predictions, test_categories):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_categories[i]:
            correct += 1
    return correct / len(predictions)


def main(number_of_k, train_filename, test_filename):
    train_features, train_categories = read_file(train_filename)
    test_features, test_categories = read_file(test_filename)

    predictions = classify(number_of_k, train_features, train_categories, test_features)
    print(predictions)

    # Calculate accuracy
    acc = accuracy(predictions, test_categories)
    print("Accuracy:", acc)

    # User Interface for inputting single vectors
    while True:
        user_input = input("\nEnter a new vector to classify (or 'exit' to stop): ")
        if user_input.lower() == 'exit':
            break

        vector = user_input.split(',')
        user_test_instance = []
        for i in range(len(vector)):
            user_test_instance.append(float(vector[i]))
        predictions = classify(number_of_k, train_features, train_categories, [user_test_instance])
        print(f'Predicted Class: {predictions[0]}')


if __name__ == '__main__':
    number_of_k = int(input("Enter the number of neighbours to be checked: "))
    train_filename = input("Enter the filename of the train data: ")
    test_filename = input("Enter the filename of the test data: ")

    # number_of_k = 3
    # train_filename = "iris.data"
    # test_filename = "iris.test.data"

    main(number_of_k, train_filename, test_filename)
