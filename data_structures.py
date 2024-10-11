# Data Structures Cheat Sheet

# 1. Arrays (Lists)
# Dynamic array implementation, can resize automatically.
array = []
array.append(1)       # Add element at the end
array.remove(1)       # Remove element
first = array[0] if array else None   # Access first element
last = array[-1] if array else None   # Access last element
length = len(array)   # Get length

# Traversal
for item in array:
    print(item)

# 2. Strings
# Immutable sequence of characters.
string = "hello"
list_string = list(string)        # Convert string to list
joined_string = ''.join(list_string)  # Convert list back to string
char = string[0]                 # Access first character
length = len(string)             # Get length
uppercase = string.upper()       # Convert to uppercase
lowercase = string.lower()       # Convert to lowercase

# Traversal
for char in string:
    print(char)

# 3. Linked Lists
# Singly Linked List: Each node points to the next.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Create a linked list: 1 -> 2 -> 3
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node1.next = node2
node2.next = node3

# Traversal
current = node1
while current:
    print(current.val)
    current = current.next

# 4. Stacks (LIFO - Last In, First Out)
# Implemented using lists or linked lists.
stack = []
stack.append(1)  # Push onto stack
stack.append(2)
stack.append(3)

top = stack.pop()  # Pop from stack (LIFO)
print(top)        # Output: 3

# 5. Queues (FIFO - First In, First Out)
# Can be implemented with lists or deque from the collections module.
from collections import deque
queue = deque()
queue.append(1)   # Enqueue (add to back)
queue.append(2)
queue.append(3)

front = queue.popleft()  # Dequeue (remove from front)
print(front)            # Output: 1

# 6. Hashmaps (Dictionaries)
# Key-value pairs with average O(1) time complexity for lookups.
dictionary = {}
dictionary["a"] = 1      # Add key-value pair
value = dictionary.get("a")  # Get value by key
keys = dictionary.keys()  # Get all keys
values = dictionary.values()  # Get all values

# 7. Sets
# Collection of unique elements, no duplicates allowed.
my_set = set()
my_set.add(1)    # Add element
my_set.add(2)
my_set.add(2)    # Duplicate elements are not added

print(my_set)    # Output: {1, 2}

# 8. Trees
# Binary Trees: Each node has at most two children.
# Binary Search Trees (BST): Left child < parent < right child.
# Balanced Trees: AVL Trees, Red-Black Trees (self-balancing).
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Create a binary tree: 1 -> (2, 3)
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)

# In-order traversal (Left, Root, Right)
def inorder_traversal(node):
    if node:
        inorder_traversal(node.left)
        print(node.val)
        inorder_traversal(node.right)

inorder_traversal(root)

# 9. Graphs
# Represented as adjacency lists or matrices.
# Can be directed or undirected.
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0],
    3: [1]
}

# Traversal using BFS (Breadth First Search)
def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node)
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

bfs(graph, 0)

# 10. Heaps
# Min Heap: Parent nodes are less than or equal to their children.
# Max Heap: Parent nodes are greater than or equal to their children.
import heapq
heap = []
heapq.heappush(heap, 3)  # Push onto heap
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

smallest = heapq.heappop(heap)  # Pop smallest element
print(smallest)  # Output: 1

# 11. Tries (Prefix Trees)
# Efficient for searching and storing strings.
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

# Example usage:
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))  # Output: True
print(trie.search("app"))    # Output: False

# 12. Disjoint Set (Union-Find)
# Used for maintaining a collection of non-overlapping sets, supporting union and find operations.
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

# Example usage:
ds = DisjointSet(5)
ds.union(0, 1)
ds.union(1, 2)
print(ds.find(0))  # Output: 0
print(ds.find(2))  # Output: 0

# 13. Segment Trees
# Useful for range query problems (e.g., sum, minimum).
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        self.build(data)

    def build(self, data):
        # Build the tree
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def range_sum(self, left, right):
        # Query the sum in the range [left, right)
        left += self.n
        right += self.n
        sum = 0
        while left < right:
            if left & 1:  # Left is odd
                sum += self.tree[left]
                left += 1
            if right & 1:  # Right is odd
                right -= 1
                sum += self.tree[right]
            left //= 2
            right //= 2
        return sum

# Example usage:
data = [1, 2, 3, 4, 5]
seg_tree = SegmentTree(data)
print(seg_tree.range_sum(1, 4))  # Output: 9 (2 + 3 + 4)

# 14. Fenwick Tree (Binary Indexed Tree)
# Provides efficient methods for cumulative frequency tables.
class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        sum = 0
        while index > 0:
            sum += self.tree[index]
            index -= index & -index
        return sum

# Example usage:
fenwick_tree = FenwickTree(5)
fenwick_tree.update(1, 1)  # Increment index 1 by 1
fenwick_tree.update(2, 2)  # Increment index 2 by 2
print(fenwick_tree.query(2))  # Output: 3 (1 + 2)

# 15. Bit Arrays
# Efficient storage of bits.
class BitArray:
    def __init__(self, size):
        self.size = size
        self.array = [0] * ((size + 31) // 32)  # Using 32-bit integers

    def set(self, index):
        if 0 <= index < self.size:
            self.array[index // 32] |= (1 << (index % 32))  # Set the bit at index to 1

    def clear(self, index):
        if 0 <= index < self.size:
            self.array[index // 32] &= ~(1 << (index % 32))  # Set the bit at index to 0

    def get(self, index):
        if 0 <= index < self.size:
            return (self.array[index // 32] >> (index % 32)) & 1  # Return the value of the bit at index
        return None  # Out of bounds

# Example usage:
bit_array = BitArray(100)
bit_array.set(5)    # Set the bit at index 5
print(bit_array.get(5))  # Output: 1
bit_array.clear(5)  # Clear the bit at index 5
print(bit_array.get(5))  # Output: 0

# 16. Bloom Filters
# Space-efficient probabilistic data structure for membership testing.
import hashlib

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * ((size + 31) // 32)

    def _hashes(self, item):
        # Generate hash values for the item
        hashes = []
        for i in range(self.num_hashes):
            hash_value = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16)
            hashes.append(hash_value % self.size)
        return hashes

    def add(self, item):
        # Add an item to the bloom filter
        for hash_value in self._hashes(item):
            self.bit_array[hash_value // 32] |= (1 << (hash_value % 32))

    def contains(self, item):
        # Check if an item is in the bloom filter
        for hash_value in self._hashes(item):
            if (self.bit_array[hash_value // 32] >> (hash_value % 32)) & 1 == 0:
                return False
        return True  # If all bits are set, item might be in the filter

# Example usage:
bloom_filter = BloomFilter(size=1000, num_hashes=5)
bloom_filter.add("apple")
bloom_filter.add("banana")

print(bloom_filter.contains("apple"))  # Output: True
print(bloom_filter.contains("banana"))  # Output: True
print(bloom_filter.contains("orange"))  # Output: False (may return True due to false positives)
