import heapq  # Import heapq for priority queue operations
import collections


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None  # Left child
        self.right = None  # Right child

    # Comparison operator, implicitly used by heapq to arrange nodes in the correct order
    def __lt__(self, other):
        return self.freq < other.freq


def calculate_frequencies(text):
    # Calculate the frequency of each character in the text
    return collections.Counter(text)  # return a dictionary with characters as keys


def build_huffman_tree(frequencies):
    # Create a priority queue with nodes created from characters and their frequencies
    priority_queue = [Node(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(priority_queue)  # Convert list into a heap

    # While there are more than one node in the queue, merge the two nodes with the lowest frequency
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)

    return priority_queue[0]  # The root node


def generate_codes(node, prefix="", code_dict={}):
    if node is not None:
        if node.char is not None:
            code_dict[node.char] = prefix  # Assign code to character
        generate_codes(node.left, prefix + "0", code_dict)  # Recur left
        generate_codes(node.right, prefix + "1", code_dict)  # Recur right
    return code_dict


def huffman_coding(text):
    frequencies = calculate_frequencies(text)
    root = build_huffman_tree(frequencies)
    codes = generate_codes(root)
    return codes


text = input("Enter the text for Huffman encoding: ")  # Example string: AAABBBBCDDEEFFFFFFFFFFFG
codes = huffman_coding(text)
print("Huffman Codes:")
for char, code in codes.items():
    print(f"'{char}': {code}")