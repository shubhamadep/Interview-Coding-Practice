'''
Sum of left nodes

'''

class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:

        def helper(root, isLeft):

            if not root:
                return 0

            if root.left is None and root.right is None:
                return root.val if isLeft else 0

            return helper(root.left, True) + helper(root.right, False)

        return helper(root, False)


'''
Left side view and right side view
BFS

'''
class Solution:
    def leftSide(self, root):
        if root is None:
            return []

        next_level = deque([root, ])
        rightside = [root.val]

        while next_level:
            # prepare for the next level
            curr_level = next_level
            next_level = deque()

            while curr_level:
                node = curr_level.popleft()

                # add child nodes of the current level
                # in the queue for the next level
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)

            # The current level is finished.
            # Its last element is the rightmost one.
            if next_level:
                rightside.append(next_level[0].val)

        return rightside

    class Solution:
        def rightSideView(self, root: TreeNode) -> List[int]:
            if root is None:
                return []

            next_level = deque([root, ])
            rightside = []

            while next_level:
                # prepare for the next level
                curr_level = next_level
                next_level = deque()

                while curr_level:
                    node = curr_level.popleft()

                    # add child nodes of the current level
                    # in the queue for the next level
                    if node.left:
                        next_level.append(node.left)
                    if node.right:
                        next_level.append(node.right)

                # The current level is finished.
                # Its last element is the rightmost one.
                rightside.append(node.val)

            return rightside

'''
Top view

'''


class Solution:
    def TopSideView(self, root: TreeNode) -> List[int]:
        if root is None:
            return []

        next_level = deque([(root, 0), ])
        result = []
        seen = set()

        while next_level:
            # prepare for the next level
            curr_level = next_level
            next_level = deque()

            while curr_level:
                node, hd = curr_level.popleft()

                if hd not in seen:
                    result.append((node.val, hd))
                    seen.add(hd)

                # add child nodes of the current level
                # in the queue for the next level
                if node.left:
                    next_level.append((node.left, hd - 1))
                if node.right:
                    next_level.append((node.right, hd + 1))

        result = sorted(result, key=lambda x: x[1])
        return [x[0] for x in result]

'''
Boundaries of binary tree
boundaries for binary tree

Time: O(n)
Space: O(n)

'''
Class Solution
    def boundaries(self, root):

        self.result = []
        t = root

        #left boundaries
        while t is not None:
            if not self.isLeaf(t):
                self.result.append(t)
            if t.left != None:
                t = t.left
            elif t.right != None:
                t = t.right

        #right boundaries
        t = root.right
        stack = []
        while t is not None:
            if not self.isLeaf(t):
                stack.push(t.val)
            if t.right != None:
                t = t.right
            elif t.left != None:
                t = t.left

        while stack:
           self. result.append(stack.pop())

        #add leaves
        self.addLeaves(root)


    def addLeaves(self, root):

        while root is not None:
            if self.isLeaf(root):
                self.result.append(root.val)
            else:
                if root.left:
                    self.addLeaves(root.left)
                if root.right:
                    self.addLeaves(root.right)

    def isLeaf(self, node):
        return node.left == None and node.right == None

'''
Validate BST

Time: O(n)
Space: O(logn)
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:

        def helper(node, min, max):
            if not node:
                return True

            if node.val >= max or node.val <= min:
                return False

            left = helper(node.left, min, node.val)

            return left and helper(node.right, node.val, max)

        return helper(root, float('-inf'), float('inf'))

'''
Top view, bottom view, left view, right view of binary tree

Solution: https://www.geeksforgeeks.org/print-nodes-top-view-binary-tree/

Top and bottom view use HD ( horizontal distances ) . And all 4 algorithms use BFS traversal

'''

'''
Zig Zag traverse of binary tree

https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/submissions/

'''

import collections

class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        q = collections.deque()
        zig = True
        q.append(root)
        result = []

        while q:

            children = []
            while q:
                children.append(q.popleft())

            if zig:
                result.append([child.val for child in children])
            else:
                result.append([child.val for child in children[::-1]])

            q = collections.deque()
            for child in children:

                if child.left:
                    q.append(child.left)
                if child.right:
                    q.append(child.right)

            zig = not zig

        return result

'''

Convert Binary tree to Doubly Linked List
Time: O(N)
Space: O(N)

'''

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""


class Solution(object):
    def treeToDoublyList(self, root):

        self.last = None
        self.first = None

        def helper(root):
            if not root:
                return None

            helper(root.left)

            if self.last:
                self.last.right = root
                root.left = self.last
            else:
                self.first = root

            self.last = root

            helper(root.right)

            return root

        helper(root)

        self.first.left = self.last
        self.last.right = self.first

        return self.first


# This is a non-recursive approach

    arr = []
    def trav(root):
        if root:
            trav(root.left)
            arr.append(root)
            trav(root.right)

    def connect(arr):
        if len(arr) <= 1:
            if not arr: return root
            arr[0].left = arr[0].right = arr[0]
            return arr[0]
        for i in range(len(arr)-1):
            arr[i].right = arr[i+1]
            arr[i].left = arr[i-1]
        j = i+1
        arr[j].right = arr[-j-1]
        arr[j].left = arr[j-1]
        return arr[0]

    trav(root)
    return connect(arr)

'''
Check if two trees are the same

Time: O(N)
Space: O(logn)

'''

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:

        def check(p, q):
            if not p and not q:
                return True

            if not p or not q:
                return False

            if p.val != q.val:
                return False

            return check(p.left, q.left) and check(p.right, q.right)

        return check(p, q)

'''
Symmetric trees

'''

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True

        def helper(left, right):
            if not left and not right:
                return True

            if not left or not right:
                return False

            return (left.val == right.val) and helper(left.left, right.right) and helper(left.right, right.left)

        return helper(root.left, root.right)

'''
height of binary tree

Time: O(N)
Space: O(logn)

'''

class Solution:
    def maxDepth(self, root: TreeNode, height=0) -> int:
        if not root:
            return 0

        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

'''

minimum height of the binary tree
Time: O(N)
Space: O(logn)

'''

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        if not root.left and not root.right:
            return 1

        ld, rd = float('inf'), float('inf')

        if root.left:
            ld = self.minDepth(root.left)
        if root.right:
            rd = self.minDepth(root.right)

        return min(ld, rd) + 1

'''
Maximum path sum 

'''


class Solution:
    def maxPathSum(self, root: TreeNode) -> int:

        if not root.left and not root.right:
            return root.val

        self.mx = float('-inf')

        def helper(root):
            if not root:
                return 0
            #this is for the case [2, -1] Answer should be 2 not 1. We do not return negative numbers.
            leftSum = max(helper(root.left), 0)
            rightSum = max(helper(root.right), 0)

            self.mx = max(self.mx, leftSum + root.val + rightSum)

            return root.val + max(leftSum, rightSum)

        helper(root)
        return self.mx

'''
https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/submissions/

'''


class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':

        if not root:
            return root

        first, last = None, None

        def helper(node):
            nonlocal first, last
            if not node:
                return

            helper(node.left)

            if last:
                last.right = node
                node.left = last
            else:
                first = node

            last = node
            helper(node.right)

        helper(root)
        first.left = last
        last.right = first

        return first

'''
Find number of nodes in complete binary tree

Time: O(d^2) d - depth ( O(logn) )
Space: O(1)

'''

#this is the brute force solution O(n)
def countNodes(self, root: TreeNode) -> int:
    if not root:
        return 0

    q = collections.deque()
    q.append(root)
    nodes = 0

    while q:
        node = q.pop()
        nodes += 1
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)

    return nodes

# optimized solution is VVV important : uses binary search in binary search

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0

        depth = self.getDepth(root)
        if depth == 0:
            return 1

        left, right = 1, 2 ** depth - 1
        while left <= right:
            pivot = left + (right - left) // 2  # ( left + right ) // 2 ; standard trick to avoid overflow
            if self.exists(pivot, depth, root):
                left = pivot + 1
            else:
                right = pivot - 1

        return (2 ** depth - 1) + left

    def getDepth(self, root):
        d = 0
        while root.left:
            root = root.left
            d += 1

        return d

    def exists(self, idx, depth, node):

        left, right = 0, 2 ** depth - 1
        for _ in range(depth):
            pivot = left + (right - left) // 2
            if idx <= pivot:
                node = node.left
                right = pivot - 1
            else:
                node = node.right
                left = pivot + 1

        return node is not None

'''
Check if the tree is balanced: All leaf nodes are on the same level or -1 and +1 level

*Remember* concept. 

Brute force algo, calculate left height and right height at every node and verify:
Time : O ( nlogn )
Space: O( n )

This algorithm:
Time complexity: O(N) 
space: O(N) -- recursive stack.

'''

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        def helper(root):
            if not root:
                return True, -1

            leftBalanced, leftDepth = helper(root.left)
            if not leftBalanced:
                return False, 0
            rightBalanced, rightDepth = helper(root.right)
            if not rightBalanced:
                return False, 0

            return (abs(leftDepth - rightDepth) < 2, 1 + max(leftDepth, rightDepth))

        return helper(root)[0]

'''

Serialize Deserialize binary tree
Time: O(n)
Space: O(n)

'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):

        if not root:
            return 'X'

        leftTree = self.serialize(root.left)
        rightTree = self.serialize(root.right)

        return str(root.val) + ',' + leftTree + ',' + rightTree

    def deserialize(self, data):

        self.q = collections.deque(data.split(','))

        def helper():
            if not self.q:
                return None

            val = self.q.popleft()
            if val == 'X':
                return None

            node = TreeNode(val)

            node.left = helper()
            node.right = helper()

            return node

        return helper()

'''

Diameter of binary tree
Time: O(N)
Space: O(1)

'''

def diameterOfBinaryTree(self, root: TreeNode) -> int:
    self.answer = 0

    def helper(root):
        if not root:
            return 0

        left = helper(root.left)
        right = helper(root.right)
        self.answer = max(self.answer, left + right)

        return max(left, right) + 1

    helper(root)


'''
Find K distant nodes from node in binary tree

Two Solutions :
1. create dictionary of parent pointers, find node, and then perform bfs.
2. find node, find k distant nodes from node in subtree, and then k-l distant nodes from root. ( recursive code ) VV important

'''


# This is an input class. Do not edit.
class BinaryTree:
    def __init__(self, value, left=None, right=None):
        self.value = value

    self.left = left
    self.right = right


def findNodesDistanceK(tree, target, k):
    nodes = []
    findDistanceFromNodeToTarget(tree, target, k, nodes)
    return nodes


def findDistanceFromNodeToTarget(node, target, k, nodes):
    if not node:
        return -1
    elif node.value == target:
        addSubtreeNodes(node, 0, k, nodes)
        return 1
    else:
        leftDistance = findDistanceFromNodeToTarget(node.left, target, k, nodes)
        rightDistance = findDistanceFromNodeToTarget(node.right, target, k, nodes)

        if leftDistance == k or rightDistance == k:
            nodes.append(node.value)

        if leftDistance != -1:
            addSubtreeNodes(node.right, leftDistance + 1, k, nodes)
            return leftDistance + 1
        if rightDistance != -1:
            addSubtreeNodes(node.left, rightDistance + 1, k, nodes)
            return rightDistance + 1
    return -1


def addSubtreeNodes(node, distance, k, nodes):
    if not node:
        return

    if distance == k:
        nodes.append(node.value)
        return

    addSubtreeNodes(node.left, distance + 1, k, nodes)
    addSubtreeNodes(node.right, distance + 1, k, nodes)



'''
max depth of n-ary tree

'''

class Solution:
    def maxDepth(self, root: 'Node') -> int:

        def find_depth(node, depth):
            if not node:
                return 0

            if node.children == []:
                return 1

            heights = [find_depth(child, depth + 1) for child in node.children]

            return max(heights) + 1

        return find_depth(root, 0)

'''
https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/

'''


class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:

        def dfs(node):
            if not node:
                return 0

            left = dfs(node.left) + 1
            right = dfs(node.right) + 1
            if node.left and node.left.val != node.val + 1:
                left = 1
            if node.right and node.right.val != node.val + 1:
                right = 1
            self.longest_consecutive_sequence = max(self.longest_consecutive_sequence, max(left, right))
            return max(left, right)

        self.longest_consecutive_sequence = 0
        dfs(root)

        return self.longest_consecutive_sequence


'''
Vertical traversal 

'''


class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        seen = collections.defaultdict(list)
        horizontal_distance = 0
        queue = collections.deque()
        depth = 0
        queue.append((root, depth, horizontal_distance))

        while queue:
            node, d, hd = queue.pop()
            seen[hd].append([d, node.val])

            if node.left:
                queue.append((node.left, d + 1, hd - 1))
            if node.right:
                queue.append((node.right, d + 1, hd + 1))

        keys = sorted(seen.keys())
        result = []
        for k in keys:
            values = seen[k]
            result.append([val for row, val in sorted(seen[k])])

        return result

'''
Generate binary tree from preorder and postorder traversals

'''


class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        if not pre:
            return None
        node = TreeNode(pre[0])
        if len(pre) == 1:
            return node

        L = post.index(pre[1]) + 1
        print(pre[1:L + 1], post[:L])
        print(pre[L + 1:], post[L:-1])
        print()
        node.left = self.constructFromPrePost(pre[1:L + 1], post[:L])
        node.right = self.constructFromPrePost(pre[L + 1:], post[L:-1])
    return node

'''
Optimized 

'''


class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:

        def make(i0, i1, N):
            if N == 0: return None
            root = TreeNode(pre[i0])
            if N == 1: return root

            for L in range(N):
                if post[i1 + L - 1] == pre[i0 + 1]:
                    break

            root.left = make(i0 + 1, i1, L)
            root.right = make(i0 + L + 1, i1 + L, N - 1 - L)
            return root

        return make(0, 0, len(pre))

'''
Recover binary tree
https://leetcode.com/problems/recover-binary-search-tree/

'''


class Solution:
    def recoverTree(self, root: TreeNode) -> None:

        stack = []
        node = root
        prev = TreeNode(float('-inf'))
        replace = []

        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            temp = stack.pop()
            if temp.val < prev.val:
                replace.append((prev, temp))
            prev = temp
            node = temp.right

        replace[0][0].val, replace[-1][1].val = replace[-1][1].val, replace[0][0].val

'''
Populating next right pointers in each node
https://leetcode.com/problems/populating-next-right-pointers-in-each-node/

'''
class Solution:
    def connect(self, node: 'Node') -> 'Node':

        def rst(root, parent, isLeft):
            if not root:
                return

            left, right = root.left, root.right

            rst(left, root, True)

            if parent is None:
                root.next = None
            elif isLeft:
                root.next = parent.right
            else:
                if parent.next is None:
                    root.next = None
                else:
                    root.next = parent.next.left

            rst(right, root, False)

        rst(node, None, None)
        return node

'''
Maximum average tree
https://leetcode.com/problems/maximum-average-subtree/submissions/

'''

class Solution:
    def maximumAverageSubtree(self, root: TreeNode) -> float:
        def dfs(node):
            if not node:
                return 0, 0
            left_count, left_sum = dfs(node.left)
            right_count, right_sum = dfs(node.right)
            self.average = max(self.average, (left_sum + node.val + right_sum) / (left_count + right_count + 1))
            return left_count + right_count + 1, left_sum + right_sum + node.val

        self.average = 0
        dfs(root)
        return self.average