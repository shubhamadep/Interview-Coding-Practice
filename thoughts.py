'''

https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station-set-2-map-based-approach/

Input:  arr[]  = {9:00,  9:40, 9:50,  11:00, 15:00, 18:00}
        dep[]  = {9:10, 12:00, 11:20, 11:30, 19:00, 20:00}
Output: 3

sort by arrivals
if departure of 1 >= arrival of next: platforms++


https://practice.geeksforgeeks.org/problems/reverse-array-in-groups0255/1

Input:
N = 5, K = 3
arr[] = {1,2,3,4,5}
Output: 3 2 1 5 4
Explanation: First group consists of elements
1, 2, 3. Second group consists of 4,5.

'''

class Solution:
    def reverse(self, array, k):
        if not array:
            return []

        if len(array) < k:
            return sorted(array)

        start, end = 0, k
        while end < len(array) and start < len(array):
            i, j = start, end
            while i < j:
                self.swap(i, j)
                i += 1
                j -= 1
            start = end + 1
            end = len(array)-1 if end + k >= len(array) else end + k

    def swap(self, i, j):
        pass

'''

https://leetcode.com/problems/reverse-nodes-in-k-group/submissions/

https://www.geeksforgeeks.org/find-pythagorean-triplet-in-an-unsorted-array/

'''

'''

https://www.geeksforgeeks.org/convert-array-into-zig-zag-fashion/

Arr[] = {4, 3, 7, 8, 6, 2, 1}
Output: 3 7 4 8 2 6 1
Explanation: 3 < 7 > 4 < 8 > 2 < 6 > 1

sort = {1, 2, 3, 4, 6, 7, 8}
       {1, 3, 2, 6, 4, 8, 7}

'''


def zigZag(arr, n):
    # Flag true indicates relation "<" is expected,
    # else ">" is expected. The first expected relation
    # is "<"
    flag = True
    for i in range(n - 1):
        # "<" relation expected
        if flag is True:
            # If we have a situation like A > B > C,
            # we get A > B < C
            # by swapping B and C
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                # ">" relation expected
        else:
            # If we have a situation like A < B < C,
            # we get A < C > B
            # by swapping B and C
            if arr[i] < arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
        flag = bool(1 - flag)
    print(arr)


# Driver program
arr = [4, 3, 7, 8, 6, 2, 1]
n = len(arr)
zigZag(arr, n)

'''
https://practice.geeksforgeeks.org/problems/largest-number-formed-from-an-array1117/1

'''

class Solution:
    def mergeSort(self, nums):
        if len(nums) == 1:
            return nums

        mid = (0+len(nums)) // 2
        left = nums[:mid]
        right = nums[mid:]

        return self.merge(self.mergeSort(left), self.mergeSort(right))

    def merge(self, left, right):
        sortedArray = []

        sortedArrayPointer = 0
        leftArrayPointer, rightArrayPointer = 0, 0

        while leftArrayPointer < len(left) and rightArrayPointer < len(right):
            if left[leftArrayPointer] <= right[rightArrayPointer]:
                sortedArray[sortedArrayPointer] = left[leftArrayPointer]
                leftArrayPointer += 1
            else:
                sortedArray[sortedArrayPointer] = right[rightArrayPointer]
                rightArrayPointer += 1

            sortedArrayPointer += 1

        while leftArrayPointer < len(left):
            sortedArray[sortedArrayPointer] = left[leftArrayPointer]

        while rightArrayPointer < len(right):
            sortedArray[sortedArrayPointer] = right[rightArrayPointer]

        return sortedArray

'''
sum of left leaves

'''

class Solution:
    def sumofLeftLeaves(self, node):
        if not node.left and not node.right:
            return node.value
        if node.left:
            return node.value + self.sumofLeftLeaves(node.left)
        else:
            return node.value + self.sumofLeftLeaves(node.right)

'''
boundaries of binary tree

'''

class Solution:
    def leftBoundary(self, node):

        result = []

        while node:
            if self.isLeft(node):
                result.append(node.value)
            if node.left:
                node = node.left
            elif node.right:
                node = node.right

        return result

    def isLeft(self, node):
        return node.left and node.right

    def topBoundary(self, node):
        if not node:
            return node

        hd = 0
        q = [[node, 0]]
        memo = {}

        while q:
            node, current_hd = q[0]
            if current_hd in memo:
                memo[current_hd] = node.value
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            q.pop(0)

        return sorted(memo)


'''
https://leetcode.com/problems/populating-next-right-pointers-in-each-node/

'''

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root

        leftmost = root
        while leftmost.left:

            head = leftmost
            while head:
                head.left.next = head.right
                if head.next:
                    head.right.next = head.next.left
                head = head.next
            leftmost = leftmost.left

        return root


'''
https://practice.geeksforgeeks.org/problems/connect-nodes-at-same-level/1

'''
class Solution:
    def connectNodes(self, node):

        if not node:
            return node

        leftmost = head
        while leftmost:

            head = leftmost
            while head:
                head.left.next = head.right
                if head.next:
                    head.right.next = head.next.left
                head = head.next
            leftmost = leftmost.left

        return node


class Solution:
    def identical(self, node1, node2):

        if node1 is not node2:
            return False
        if not node1 and node2:
            return False
        if node1 and not node2:
            return False

        return self.identical(node1.left, node2.left) and self.identical(node1.right, node2.right)


class Solution:
    def symmetric(self, node):
        if not node:
            return node

        return self.helper(node.left, node.right)

    def helper(self, node1, node2):
        if node1.val != node2.val:
            return False
        return self.helper(node1.left, node2.left) and self.helper(node1.right, node2.right)


class Solution:
    def racers(self, lists):

        scores = {1: 10, 2: 6, 3: 4, 4: 3, 5: 2, 6: 1}

        for i in range(len(scores)):
            scores[i][2] = scores[scores[i][2]]

        racers = {}
        for i in range(len(lists)):
            racer = lists[i]
            if racer in racers:
                racers[racer] += racer[2]
            else:
                racers[racer] = racer[2]

        #sort and get maximum
        sorted_list = []

        maximum_score = float('inf')
        for score in sorted_list:
            maximum_score = max(score[2], maximum_score)


        result = []
        for score in sorted_list:
            if score[2] == maximum_score:
                result.append(score)

        return result


'''
validate binary search problem

'''
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True

        minimum, maximum = float('float'), float('-inf')
        return isValid(root, minimum, root.val) and isValid(root, root.value, maximum)

    def isValid(self, node, minimum, maximum):
        if not node:
            return True
        if node.val > maximum and node.val < minimum:
            return False
        return isValid(node, minimum, root.val) and isValid(node, root.value, maximum)


'''
mirror tree

'''
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        return symmetric(root.left, root.right)

    def symmetric(self, leftnode, rightnode):
        if not leftnode and not rightnode:
            return True
        if (leftnode and not rightnode) or (rightnode and not leftnode):
            return False

        return (leftnode.value == rightnode.value) and symmetric(leftnode.left, rightnode.right) and symmetric(leftnode.right, rightnode.left)


'''
count distinct integers in window of size k 

'''
class Solution:
    def distinct(self, nums, k):
        if len(nums) < k:
            return []

        left, right = 0, 0
        result = []
        distinctNumbers = 0
        seen = {}

        for i in range(k):
            if nums[i] not in seen:
                seen[nums[i]] = 1
                distinctNumbers += 1
            else:
                seen[nums[i]] += 1
        result.append(distinctNumbers)

        for i in range(k, len(nums)):
            left = nums[i-k]
            if seen[nums[left]] == 1:
                distinctNumbers -= 1
            seen[nums[left]] -= 1

            if nums[i] not in seen:
                distinctNumbers += 1
                seen[nums[i]] = 1
            else:
                seen[nums[i]] += 1

            result.append(distinctNumbers)

        return result

class Solution:
    def smallest_window(self, string, patt):
        start = 0
        count_required = 0
        count_chars_patt = {}
        minimum_window = float('inf')

        for c in patt:
            if c in count_chars_patt:
                count_chars_patt[c] += 1
            else:
                count_chars_patt[c] = 0

        for end in range(len(string)):
            if string[end] in count_chars_patt:
                if count_chars_patt[string[end]] > 0:
                    count_required += 1
                count_chars_patt[string[end]] -= 1

            while count_required == len(patt):
                if end - start + 1 > minimum_window:
                    minimum_window = end - start + 1
                    result = string[start: end+1]

                if string[start] in count_chars_patt:
                    count_chars_patt[string[start]] += 1
                    if count_chars_patt[string[start]] > 0:
                        count_required -= 1
                start += 1

        return result



'''
Implement Stack 

'''

class Stack:

    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.stack.pop()

    def peek(self):
        return self.stack[len(self.stack) - 1]

    def get_min(self):
        pass


class Solution:
    def solution(self, nums):

        def helper(nums, points, nums_count, seen):
            if len(seen) == len(nums_count):
                return 0

            for num in nums:
                temp_points = 0
                if num not in seen:
                    temp_points += nums_count[num] * num
                    seen.add(num)
                    if num - 1 in nums_count:
                        temp_points += nums_count[num-1] * (num-1)
                        seen.add(num - 1)
                    if num + 1 in nums_count:
                        temp_points += nums_count[num+1] * (num+1)
                        seen.add(num + 1)

                    self.maximum_points = max(self.maximum_points, helper(nums.remove(num),points + temp_points, nums_count, seen))

        self.maximum_points = float('inf')
        nums_count = {}
        for num in nums:
            if num in nums:
                nums_count[num] += 1
            else:
                nums_count[num] = 1

        helper(nums, 0, nums_count, set())
        return self.maximum_points


class UnionFind:

    def __init__(self, N):
        self.parent = [i for i in range(N)]
        self.components = N
        self.size = [1] * N

    def union(self, p, q):
        p_parent = self.find(p)
        q_parent = self.find(q)

        if p_parent == q_parent:
            return

        if self.size[p_parent] > self.size[q_parent]:
            self.parent[q_parent] = p
            self.size[p_parent] += self.size[q_parent]
        else:
            self.parent[p_parent] = q
            self.size[q_parent] += self.size[p_parent]
        self.components -= 1

    def find(self, p):
        node = p
        while node != self.parent[node]:
            node = self.parent[node]
        while p != node:
            p, self.parent[node] = self.parent[node], node
        return node

    def parent(self, node):
        return self.array[node]

    def isConnected(self, p, q):
        return self.parent[p] == self.parent[q]

'''
Heap Class

        5
    4       3
2      1 2      6

[5 4 3 2 1 2 6]
2*index+1, 2*index+2

'''
class Heap:
    def __init__(self):
        self.heap = []
        self.length = 0

    def push(self, value):
        self.heap.append(value)
        self.length += 1
        self.siftUp(self.length-1)

    def pop(self):
        self.swap(self.length-1, 0, self.heap)
        value = self.heap.pop()
        self.siftDown(0, self.length)
        return value

    def siftUp(self, index):
        parent = (index // 2)-1
        while parent:
            if self.heap[parent] <= self.heap[index]:
                self.swap(index, parent)
            parent = (parent // 2) - 1

    def siftDown(self):
        pass

    def peak(self):
        return self.heap[0]

    def swap(self, x, y, array):
        array[x], array[y] = array[y], array[x]
