# Data Structures Cheat Sheet

# 1. Arrays (Lists)
# Dynamic array implementation, can resize automatically.
# Time Complexity:
# - Access: O(1) - Direct access to any element using index.
# - Search: O(n) - Linear search in the worst case.
# - Insertion: O(n) - Inserting an element (worst case: when resizing is needed).
# - Deletion: O(n) - Deleting an element (shifting elements).
# Space Complexity: O(n) - Requires space for n elements.

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
# Time Complexity:
# - Access: O(1) - Direct access to any character using index.
# - Search: O(n) - Linear search in the worst case.
# - Insertion: O(n) - Inserting a character requires creating a new string.
# - Deletion: O(n) - Deleting a character requires creating a new string.
# Space Complexity: O(n) - Requires space for n characters.

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
# Time Complexity:
# - Access: O(n) - Must traverse the list to access a node.
# - Search: O(n) - Linear search through the list.
# - Insertion: O(1) - Insert at the head or tail in constant time.
# - Deletion: O(1) - Delete at the head in constant time, O(n) to find and delete.
# Space Complexity: O(n)

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
# Time Complexity:
# - Push: O(1)
# - Pop: O(1)
# - Peek: O(1)
# - Search: O(n)
# Space Complexity: O(n)

stack = []
stack.append(1)  # Push onto stack
stack.append(2)
stack.append(3)

top = stack.pop()  # Pop from stack (LIFO)
print(top)        # Output: 3

# 5. Queues (FIFO - First In, First Out)
# Time Complexity:
# - Enqueue: O(1)
# - Dequeue: O(1)
# - Peek: O(1)
# - Search: O(n)
# Space Complexity: O(n)

from collections import deque
queue = deque()
queue.append(1)   # Enqueue (add to back)
queue.append(2)
queue.append(3)

front = queue.popleft()  # Dequeue (remove from front)
print(front)            # Output: 1

# 6. Hashmaps (Dictionaries)
# Time Complexity:
# - Access: O(1) average, O(n) worst case (with collisions).
# - Search: O(1) average, O(n) worst case.
# - Insertion: O(1) average, O(n) worst case.
# - Deletion: O(1) average, O(n) worst case.
# Space Complexity: O(n)

dictionary = {}
dictionary["a"] = 1      # Add key-value pair
value = dictionary.get("a")  # Get value by key
keys = dictionary.keys()  # Get all keys
values = dictionary.values()  # Get all values

# 7. Sets
# Time Complexity:
# - Access: O(1) average.
# - Search: O(1) average.
# - Insertion: O(1) average.
# - Deletion: O(1) average.
# Space Complexity: O(n)

my_set = set()
my_set.add(1)    # Add element
my_set.add(2)
my_set.add(2)    # Duplicate elements are not added

print(my_set)    # Output: {1, 2}

# 8. Trees (Binary Search Trees)
# Time Complexity:
# - Access: O(log n) in balanced trees, O(n) in unbalanced.
# - Search: O(log n) in balanced, O(n) in unbalanced.
# - Insertion: O(log n) in balanced, O(n) in unbalanced.
# - Deletion: O(log n) in balanced, O(n) in unbalanced.
# Space Complexity: O(n)

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
        print(node.val, end=" ")  # Visit the root
        inorder_traversal(node.right)

inorder_traversal(root)

# Pre-order traversal (Root, Left, Right)
def preorder_traversal(node):
    if node:
        print(node.val, end=" ")  # Visit the root
        preorder_traversal(node.left)
        preorder_traversal(node.right)

preorder_traversal(root)

# Post-order traversal (Left, Right, Root)
def postorder_traversal(node):
    if node:
        postorder_traversal(node.left)
        postorder_traversal(node.right)
        print(node.val, end=" ")  # Visit the root

postorder_traversal(root)

from collections import deque

# Level-order traversal (Root, Level 2, Level 3, ...)
def level_order_traversal(root):
    if not root:
        return
    
    queue = deque([root])
    while queue:
        node = queue.popleft()  # Dequeue the front node
        print(node.val, end=" ")  # Visit the node
        
        # Enqueue left and right children
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

level_order_traversal(root)

# 9. Graphs
# Time Complexity:
# - BFS/DFS: O(V + E), where V is vertices and E is edges.
# - Search: O(V + E)
# Space Complexity: O(V + E)

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
# Time Complexity:
# - Push: O(log n)
# - Pop: O(log n)
# - Peek: O(1)
# Space Complexity: O(n)

import heapq
heap = []
heapq.heappush(heap, 3)  # Push onto heap
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

smallest = heapq.heappop(heap)  # Pop smallest element
print(smallest)  # Output: 1

# 11. Tries (Prefix Trees)
# Time Complexity:
# - Insertion: O(m), where m is the length of the word.
# - Search: O(m)
# Space Complexity: O(m * n), where m is the average word length and n is the number of words.

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
# Time Complexity:
# - Find: O(α(n)), where α is the inverse Ackermann function.
# - Union: O(α(n))
# Space Complexity: O(n)

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
# Time Complexity:
# - Build: O(n)
# - Update: O(log n)
# - Query: O(log n)
# Space Complexity: O(n)

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        self.build(data)

    def build(self, data):
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def range_sum(self, left, right):
        left += self.n
        right += self.n
        sum = 0
        while left < right:
            if left & 1:
                sum += self.tree[left]
                left += 1
            if right & 1:
                right -= 1
                sum += self.tree[right]
            left //= 2
            right //= 2
        return sum

# 14. Fenwick Tree (Binary Indexed Tree)
# Time Complexity:
# - Update: O(log n)
# - Query: O(log n)
# Space Complexity: O(n)

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
fenwick_tree.update(1, 1)
fenwick_tree.update(2, 2)
print(fenwick_tree.query(2))  # Output: 3

# 15. Bit Arrays
# Time Complexity:
# - Access: O(1)
# - Set/Clear: O(1)
# Space Complexity: O(n) (n is the number of bits)

class BitArray:
    def __init__(self, size):
        self.size = size
        self.array = [0] * ((size + 31) // 32)  # Using 32-bit integers

    def set(self, index):
        if 0 <= index < self.size:
            self.array[index // 32] |= (1 << (index % 32))

    def clear(self, index):
        if 0 <= index < self.size:
            self.array[index // 32] &= ~(1 << (index % 32))

    def get(self, index):
        if 0 <= index < self.size:
            return (self.array[index // 32] >> (index % 32)) & 1
        return None

# 16. Bloom Filters
# Time Complexity:
# - Insertion: O(k), where k is the number of hash functions.
# - Query: O(k)
# Space Complexity: O(m), where m is the size of the bit array.

import hashlib

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * ((size + 31) // 32)

    def _hashes(self, item):
        hashes = []
        for i in range(self.num_hashes):
            hash_value = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16)
            hashes.append(hash_value % self.size)
        return hashes

    def add(self, item):
        for hash_value in self._hashes(item):
            self.bit_array[hash_value // 32] |= (1 << (hash_value % 32))

    def contains(self, item):
        for hash_value in self._hashes(item):
            if (self.bit_array[hash_value // 32] >> (hash_value % 32)) & 1 == 0:
                return False
        return True

# Example usage:
bloom_filter = BloomFilter(size=1000, num_hashes=5)
bloom_filter.add("apple")
bloom_filter.add("banana")

print(bloom_filter.contains("apple"))  # Output: True
print(bloom_filter.contains("banana"))  # Output: True
print(bloom_filter.contains("orange"))  # Output: False
