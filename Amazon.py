'''

'''

def maxItems(numGroups, arr):
    arr.sort()
    arr[0] = 1

    for i in range(1, numGroups):
        if arr[i] - arr[i-1] > 1:
            arr[i] = arr[i-1] + 1
    return arr[-1]
'''
Top K frequent words

https://leetcode.com/discuss/interview-question/542597/

'''
import heapq


class Solution:
    def topKFrequent(self, words: List[str], K: int) -> List[str]:
        heap = []
        count = collections.Counter(words)

        for key, val in count.items():
            heap.append((-val, key))
        heapq.heapify(heap)

        return [heapq.heappop(heap)[1] for _ in range(K)]

# n log n solution
#         count = collections.Counter(words)
#         candidates = list( count.keys() )
#         candidates.sort( key = lambda w: ( -count[w], w) )

#         return candidates[:K]

'''
Rotten oranges / Zombies in matrix / min hours to send files to all the servers
similar questions related to walls / servers

refer bfs.py 
https://leetcode.com/problems/rotting-oranges/

'''

'''
Product suggestions

https://leetcode.com/problems/search-suggestions-system/submissions/

'''


class Trie:
    def __init__(self):
        self.root = {}
        self.endSymbol = '*'

    def insert(self, word):
        root = self.root

        for char in word:
            if char not in root:
                root[char] = {}
            root = root[char]

        root[self.endSymbol] = word

    def findMatches(self, word):
        root = self.root

        self.result = []

        for char in word:
            if char not in root:
                return []
            root = root[char]

        self.dfs(root)

        return self.result

    def dfs(self, root):

        if type(root) == str:
            return

        if self.endSymbol in root:
            self.result.append(root[self.endSymbol])

        for key in list(root.keys()):
            self.dfs(root[key])

        return


class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:

        trie = Trie()
        result = []

        for product in products:
            trie.insert(product)

        for i in range(len(searchWord)):
            matches = trie.findMatches(searchWord[:i + 1])
            words = sorted(matches)

            result.append(words[:3])

        return result


'''
Reorder Data in Log Files

'''

class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:

        rNum, rAlpha = [], []

        for log in logs:
            print(log.split(" ")[1])
            if log.split(" ")[1].isalpha():
                rAlpha.append(log)
            else:
                rNum.append(log)

        def sortFun(word):
            contents = word.split()
            iden, values = contents[0], contents[1:]

            return " ".join(values) + iden

        rAlpha.sort(key=lambda x: sortFun(x))

        return rAlpha + rNum

'''
Partitions labels

https://leetcode.com/problems/partition-labels/

'''


class Solution:
    def partitionLabels(self, S: str) -> List[int]:

        endOcc = collections.defaultdict()
        partitions = []

        for i in range(len(S)):
            c = S[i]
            endOcc[c] = i

        p, q = 0, 0

        while p < len(S):
            char = S[p]
            q = endOcc[char]
            s = p
            while p <= q:
                char = S[p]
                eo = endOcc[char]
                if eo > q:
                    q = eo
                p += 1

            partitions.append(q - s + 1)
            p = q + 1

        return partitions

'''
Lowest Common Ancestor for binary search tree

'''


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

# iterative

        pv = p.val
        qv = q.val

        while root:
            parent = root.val
            if pv > parent and qv > parent:
                root = root.right
            elif pv < parent and qv < parent:
                root = root.left
            else:
                return root

# Recursive

        parent = root.val
        pv = p.val
        qv = q.val

        if pv > parent and qv > parent:
            return self.lowestCommonAncestor(root.right, p, q)
        elif pv < parent and qv < parent:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root

'''
Lowest common ancestor for binary tree

https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/

'''


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        stack = [root]

        # Dictionary for parent pointers
        parent = {root: None}

        # Iterate until we find both the nodes p and q
        while p not in parent or q not in parent:

            node = stack.pop()

            # While traversing the tree, keep saving the parent pointers.
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Ancestors set() for node p.
        ancestors = set()

        # Process all ancestors for node p using parent pointers.
        while p:
            ancestors.add(p)
            p = parent[p]

        # The first ancestor of q which appears in
        # p's ancestor set() is their lowest common ancestor.
        while q not in ancestors:
            q = parent[q]
        return q

#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
#         self.ans = None

#         self.dfs(root, p, q)

#         return self.ans


#     def dfs(self, root, p, q):

#         if not root:
#             return False

#         left = self.dfs( root.left, p, q )

#         right = self.dfs( root.right, p, q )

#         mid = p == root or root == q

#         if ( left and right ) or ( mid and left ) or ( mid and right ):
#             self.ans = root

#         return mid or left or right

'''
Optimal utilization

https://leetcode.com/discuss/interview-question/373202

'''
class Solution:
    def findPairs(self, a, b, target):
        a.sort(key=lambda x: x[1])
        b.sort(key=lambda x: x[1])
        l, r = 0, len(b) - 1
        ans = []
        curDiff = float('inf')
        while l < len(a) and r >= 0:
            id1, i = a[l]
            id2, j = b[r]
            if (target - i - j == curDiff):
                ans.append([id1, id2])
            elif (i + j <= target and target - i - j < curDiff):
                ans.clear()
                ans.append([id1, id2])
                curDiff = target - i - j
            if (target > i + j):
                l += 1
            else:
                r -= 1
        return ans

'''
Connecting sticks / Min Cost to Connect Ropes / Min Time to Merge Files

'''


class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        pq = sticks
        heapq.heapify(pq)
        res = 0
        while len(pq) > 1:
            i, j = heapq.heappop(pq), heapq.heappop(pq)
            res += (i + j)
            heapq.heappush(pq, i + j)

        return res

'''
Treasure Island / Treasure Island / Min Distance to Remove the Obstacle 
https://leetcode.com/discuss/interview-question/347457

'''


class TreasureMap:
    def min_route(self, grid):
        if len(grid) == 0 or len(grid[0]) == 0:
            return -1
        self.max_r, self.max_c = len(grid), len(grid[0])
        queue = collections.deque()
        queue.append((0, 0, 0))
        grid[0][0] = "D"
        while queue:
            row, col, level = queue.popleft()
            for r, c in self.getNeighbors(row, col, grid):
                if grid[r][c] == "D":
                    pass
                elif grid[r][c] == "X":
                    return level + 1
                else:
                    queue.append((r, c, level + 1))
                    grid[r][c] = "D"
        return -1

    def getNeighbors(self, row, col, grid):
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < self.max_r and 0 <= c < self.max_c:
                yield r, c

'''
Treasure Island 2

https://leetcode.com/discuss/interview-question/356150
'''


def treasure_island(A):
    row = len(A)
    if row == 0:
        return -1
    col = len(A[0])

    q = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for i in range(row):
        for j in range(col):
            if A[i][j] == "S":
                q.append([0, i, j])

    while (len(q)):
        step, r, c = q.pop(0)
        A[r][c] = 'D'

        for dr, dc in directions:
            nr = r + dr
            nc = c + dc

            if 0 <= nr < row and 0 <= nc < col:
                if A[nr][nc] == 'X':
                    return step + 1
                elif A[nr][nc] == 'O':
                    q.append([step + 1, nr, nc])

    return -1

'''
chopping gold course ( same as above ) 

https://leetcode.com/problems/cut-off-trees-for-golf-event/
'''


class Solution:
    def cutOffTree(self, forest):

        # iterate over forest and find r, c for all trees. sort them according to v.
        trees = []
        for r in range(len(forest)):
            for c in range(len(forest[0])):
                if forest[r][c] > 0:
                    trees.append((forest[r][c], r, c))
        print(sorted(trees))

        # iterate over sorted trees, and find distance between first two trees, and so on.
        trees = sorted(trees)

        sr = sc = ans = 0
        for _, tr, tc in trees:
            d = self.bfs(forest, sr, sc, tr, tc)
            if d < 0: return -1
            ans += d
            sr, sc = tr, tc

        return ans

    # classic BFS
    def bfs(self, forest, sr, sc, tr, tc):
        R, C = len(forest), len(forest[0])
        queue = collections.deque([(sr, sc, 0)])
        seen = {(sr, sc)}
        while queue:
            r, c, d = queue.popleft()
            if r == tr and c == tc:
                return d
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < R and 0 <= nc < C and
                        (nr, nc) not in seen and forest[nr][nc]):
                    seen.add((nr, nc))
                    queue.append((nr, nc, d + 1))
        return -1

'''
2 sum / movies in flight 

https://leetcode.com/discuss/interview-question/356960

'''

'''
Deep copy of Linked List with random pointer.

https://leetcode.com/problems/copy-list-with-random-pointer/
'''


class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':

        if not head:
            return None

        p = head
        while p:
            new_node = Node(p.val, None, None)
            new_node.next = p.next
            p.next = new_node
            p = new_node.next

        ptr = head
        while ptr:
            ptr.next.random = ptr.random.next if ptr.random else None
            ptr = ptr.next.next

        oldhead = head
        newhead = head.next
        newH = head.next

        while oldhead:
            oldhead.next = oldhead.next.next
            newhead.next = newhead.next.next if newhead.next else None
            oldhead = oldhead.next
            newhead = newhead.next

        return newH

'''
Merge two sorted list 
https://leetcode.com/problems/merge-two-sorted-lists/

'''


class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        if not l1:
            return l2
        if not l2:
            return l1

        p, q = l1, l2

        newhead = ListNode(-1, None)
        R = newhead

        while p and q:

            if p.val <= q.val:
                newhead.next = p
                p = p.next
            else:
                newhead.next = q
                q = q.next
            newhead = newhead.next

        if p:
            newhead.next = p
        elif q:
            newhead.next = q

        return R.next

'''
Subtree of another tree

https://leetcode.com/problems/subtree-of-another-tree/
'''


class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if not t:
            return False

        return self.check(s, t)

    def check(self, s, t):

        if not s:
            return False
        if (s.val == t.val) and self.checkSame(s, t):
            return True

        return self.check(s.left, t) or self.check(s.right, t)

    def checkSame(self, s, t):

        if not s and not t:
            return True
        elif not s or not t:
            return False
        elif s.val != t.val:
            return False
        else:
            return self.checkSame(s.left, t.left) and self.checkSame(s.right, t.right)

'''
Search 2D matrix

'''


class Solution:
    def searchMatrix(self, matrix, target):

        i, j = len(matrix) - 1, 0
        while (i >= 0 and i < len(matrix) and j >= 0 and j < len(matrix[0])):
            num = matrix[i][j]
            if num == target:
                return True
            elif num < target:
                j += 1
            elif num > target:
                i -= 1
        return False

'''
Favorite genre

https://leetcode.com/discuss/interview-question/373006

'''

'''
Spiral traverse

https://www.algoexpert.io/questions/Spiral%20Traverse
'''

'''
VVV important question to understand. 

Subarrays with K distinct integers 

https://leetcode.com/discuss/interview-question/370157


'''


class Solution:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:

        def helper(A, K):
            count = 0
            i, j = 0, 0
            distinct = {}

            while j < len(A):
                distinct[A[j]] = distinct.get(A[j], 0) + 1
                while len(distinct) > K:
                    distinct[A[i]] -= 1
                    if distinct[A[i]] == 0:
                        del distinct[A[i]]
                    i += 1
                count += j - i + 1
                j += 1

            return count

        return helper(A, K) - helper(A, K - 1)

'''
Minimum path sum 
number of unique paths
https://leetcode.com/problems/minimum-path-sum/

'''


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:

        # Dynamic programming - storing min to reach a point from   destination.
        for i in reversed(range(len(grid))):
            for j in reversed(range(len(grid[0]))):
                # elements on the bottom of the row
                if i == len(grid) - 1 and j != len(grid[0]) - 1:
                    grid[i][j] += grid[i][j + 1]
                # elements on the right column
                if j == len(grid[0]) - 1 and i != len(grid) - 1:
                    grid[i][j] += grid[i + 1][j]
                # elements not on the border.
                if j != len(grid[0]) - 1 and i != len(grid) - 1:
                    grid[i][j] += min(grid[i + 1][j], grid[i][j + 1])

        return grid[0][0]

'''
Unique paths 2
number of unique paths with obstacles

'''


class Solution:
    def uniquePathsWithObstacles(self, grid: List[List[int]]) -> int:

        if grid[0][0] == 1:
            return 0

        rows = len(grid)
        columns = len(grid[0])

        grid[0][0] = 1  # 1 path to reach the start

        for i in range(1, rows):
            grid[i][0] = int(grid[i][0] == 0 and grid[i - 1][0] == 1)

        for j in range(1, columns):
            grid[0][j] = int(grid[0][j] == 0 and grid[0][j - 1] == 1)

        for i in range(1, rows):
            for j in range(1, columns):
                if grid[i][j] == 0:
                    grid[i][j] = grid[i - 1][j] + grid[i][j - 1]
                else:
                    grid[i][j] = 0

        return grid[rows - 1][columns - 1]

'''
Max of min altitudes 

https://leetcode.com/discuss/interview-question/383669/
'''

res = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]

row = len(grid) - 1
col = len(grid[0]) - 1

for i in range(row, -1, -1):
    for j in range(col, -1, -1):
        if i == row and j == col:
            res[i][j] = float('inf')
        elif i == row and j != col:
            res[i][j] = min(grid[i][j], res[i][j + 1])

        elif i != row and j == col:
            res[i][j] = min(grid[i][j], res[i + 1][j])

        elif i == 0 and j == 0:
            res[i][j] = max(res[i + 1][j], res[i][j + 1])

        else:
            res[i][j] = max(min(grid[i][j], res[i][j + 1]), min(grid[i][j], res[i + 1][j]))

return res[0][0]

'''
K closest points to origin

'''

class Solution(object):
    def kClosest(self, points, K):
        points.sort(key = lambda P: P[0]**2 + P[1]**2)
        return points[:K]

'''
Dont worry about this question
Too mathematical
https://leetcode.com/problems/generate-parentheses/solution/
'''

'''
Prison after N days

'''


class Solution:
    def prisonAfterNDays(self, cells: List[int], N: int) -> List[int]:

        trackCombinations = {}
        combination = tuple(cells)

        while True:
            combination = self.getNextCombination(combination)
            if combination in trackCombinations:
                break
            else:
                trackCombinations[combination] = combination

        patternRepeatsAfter = (N) % len(trackCombinations)

        combinations = list(trackCombinations.values())

        return combinations[patternRepeatsAfter - 1]

    def getNextCombination(self, combination):
        cells = list(combination)

        for i in range(1, len(combination) - 1):

            if combination[i - 1] ^ combination[i + 1] == 0:
                cells[i] = 1
            else:
                cells[i] = 0

        cells[0] = 0
        cells[-1] = 0
        return tuple(cells)


'''
Min Cost to Repair Edges
Min Cost to Connect All Nodes (a.k.a. Min Cost to Add New Roads)

https://leetcode.com/discuss/interview-question/356981

https://leetcode.com/discuss/interview-question/357310
Just take the unionFindKruskalsAlgorithm and update already connected node's cost as 0.

'''

'''
Highest Profit

https://leetcode.com/discuss/interview-question/823177/Amazon-or-OA-2020-or-Find-The-Highest-Profit

'''

'''
baseball 

https://leetcode.com/problems/baseball-game/description/

'''

'''
maximum units

https://leetcode.com/discuss/interview-question/793606/Amazon-or-OA-2020-or-Maximum-Units

'''

'''
All possible lengths of substring with k - 1
largest item association 
https://leetcode.com/discuss/interview-question/783947/Amazon-or-OA-or-SDE-1-or-Aug-2020

'''
