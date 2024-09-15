def knapsack_recursive(weights, values, n, W):
    # no items left or no capacity left
    if n == 0 or W == 0:
        return 0

    # if weight of the nth item is more than capacity, don't inclue
    if weights[n-1] > W:
        return knapsack_recursive(weights, values, n-1, W)
    else:
        # If include the nth item
        include_item = values[n - 1] + knapsack_recursive(weights, values, n - 1, W - weights[n - 1])
        # If don't include the nth item
        exclude_item = knapsack_recursive(weights, values, n - 1, W)

        # Compare the two results and return the greater one
        return max(include_item, exclude_item)


weights1 = [3, 1, 6, 10, 1, 4, 9, 1, 7, 2, 6, 1, 6, 2, 2, 4, 8, 1, 7, 3, 6, 2, 9, 5, 3, 3, 4, 7, 3, 5, 30, 50]
values1 = [7, 4, 9, 18, 9, 15, 4, 2, 6, 13, 18, 12, 12, 16, 19, 19, 10, 16, 14, 3, 14, 4, 15, 7, 5, 10, 10, 13, 19, 9, 8, 5]
n1 = 32
W1 = 75

result1 = knapsack_recursive(weights1, values1, n1, W1)
print(result1)
