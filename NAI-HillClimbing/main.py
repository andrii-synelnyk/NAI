import random

# Define the weights, values, and maximum weight capacity
weights = [3, 1, 6, 10, 1, 4, 9, 1, 7, 2, 6, 1, 6, 2, 2, 4, 8, 1, 7, 3, 6, 2, 9, 5, 3, 3, 4, 7, 3, 5, 30, 50]
values = [7, 4, 9, 18, 9, 15, 4, 2, 6, 13, 18, 12, 12, 16, 19, 19, 10, 16, 14, 3, 14, 4, 15, 7, 5, 10, 10, 13, 19, 9, 8, 5]
W_max = 75  # Maximum weight the knapsack can hold

# 00011110001101101111111000111011
# total weight: 75
# total value: 262


def random_solution(weights, W_max):
    # Generate a random solution where each item may or may not be included.
    solution = [random.randint(0, 1) for _ in weights]
    while total_weight(solution, weights) > W_max:
        solution = [random.randint(0, 1) for _ in weights]
    return solution


def total_weight(solution, weights):
    # Calculate total weight of the selected items.
    return sum(w * s for w, s in zip(weights, solution))


def total_value(solution, values):
    # Calculate total value of the selected items.
    return sum(v * s for v, s in zip(values, solution))


def get_neighbors(solution):
    # Generate all possible single flip neighbors of the current solution.
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution[:]  # Create a copy, don't use the reference to the actual solution object
        neighbor[i] = 1 - neighbor[i]  # Flip the item's inclusion
        neighbors.append(neighbor)
    return neighbors


def hill_climbing(weights, values, W_max):
    # Perform the hill climbing algorithm.
    current_solution = random_solution(weights, W_max)
    current_value = total_value(current_solution, values)
    while True:
        neighbors = get_neighbors(current_solution)
        next_solution = None
        for neighbor in neighbors:
            if total_weight(neighbor, weights) <= W_max:
                neighbor_value = total_value(neighbor, values)
                if neighbor_value > current_value:
                    next_solution = neighbor
                    current_value = neighbor_value
        if next_solution is None:
            break  # No better neighbor found
        current_solution = next_solution
    return current_solution, current_value


solution, value = hill_climbing(weights, values, W_max)
print("Best value found:", value)
print("Items included:", solution)
