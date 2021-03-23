'''
Classic collections of DP problems

'''

'''
Coin change problem

'''

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:

        change = [amount + 1 for _ in range(amount + 1)]
        change[0] = 0

        for i in range(amount + 1):
            for coin in coins:
                if coin <= i:
                    change[i] = min(change[i], change[i - coin] + 1)

        return change[amount] if change[amount] <= amount else -1

'''
    
    Longest common subsequence
    
'''


def longestCommonSubsequence(str1, str2):
    dp = [[0 for _ in range(len(str1) + 1)] for _ in range(len(str2) + 1)]

    for r in range(1, len(dp)):
        for c in range(1, len(dp[0])):

            if str2[r - 1] == str1[c - 1]:
                dp[r][c] = dp[r - 1][c - 1] + 1
            else:
                dp[r][c] = max(dp[r - 1][c], dp[r][c - 1])

    return backtrack(dp, str1)


def backtrack(dp, str1):
    r, c = len(dp) - 1, len(dp[0]) - 1
    sequence = []

    while r != 0 and c != 0:
        curr, left, top, diag = dp[r][c], dp[r][c - 1], dp[r - 1][c], dp[r - 1][c - 1]
        if left == curr:
            c -= 1
        elif top == curr:
            r -= 1
        else:
            sequence.append(str1[c - 1])
            r -= 1
            c -= 1

    return sequence[::-1]



'''
Classic Knapsack problem

'''

def knapsackProblem(items, capacity):
    dp = [[0 for _ in range(capacity + 1)] for _ in range(len(items) + 1)]

    for row in range(1, len(dp)):

        weight, value = items[row - 1][1], items[row - 1][0]
        for col in range(1, len(dp[0])):

            if col - weight < 0:
                dp[row][col] = dp[row - 1][col]
            else:
                dp[row][col] = max(dp[row - 1][col], dp[row - 1][col - weight] + value)

    print(dp)
    return [dp[-1][-1], backtrack(dp, items)]


def backtrack(dp, items):
    sequence = []
    r, c = len(dp) - 1, len(dp[0]) - 1

    while r > 0:
        if dp[r][c] == dp[r - 1][c]:
            r -= 1
        else:
            sequence.append(r - 1)
            c -= items[r - 1][1]
            r -= 1
        if c == 0:
            break
    return list(reversed(sequence))



'''
Jump Game

'''


class Solution:
    def canJump(self, nums: List[int]) -> bool:

        lastPos = len(nums) - 1
        for i in reversed(range(len(nums) - 1)):
            if i + nums[i] >= lastPos:
                lastPos = i

        return lastPos == 0


'''
Jump Game II

This question can also be solved using DP with O(N) time and O(N) space

Time : O(N)
Space: O(1)

'''


class Solution:
    def jump(self, nums: List[int]) -> int:

        n = len(nums)
        if n < 2:
            return 0

        maxPos = nums[0]
        maxSteps = nums[0]
        jumps = 1

        for i in range(len(nums)):
            if maxSteps < i:
                jumps += 1
                maxSteps = maxPos
            maxPos = max(maxPos, nums[i] + i)

        return jumps


'''
Edit Distance

'''


def levenshteinDistance(str1, str2):
    # matrix with size m, n
    lenstr1, lenstr2 = len(str1), len(str2)
    editDistance = [[x for x in range(lenstr1 + 1)] for _ in range(lenstr2 + 1)]
    print(editDistance)
    for i in range(1, len(str2) + 1):
        editDistance[i][0] = editDistance[i - 1][0] + 1

    print(editDistance)
    # Then loop through the matrix from 2nd row & column
    for i in range(1, lenstr2 + 1):
        for j in range(1, lenstr1 + 1):
            print(i, j)
            if str2[i - 1] == str1[j - 1]:
                editDistance[i][j] = editDistance[i - 1][j - 1]
            else:
                editDistance[i][j] = min(editDistance[i - 1][j], editDistance[i - 1][j - 1], editDistance[i][j - 1]) + 1
            print(editDistance[i][j])
    return editDistance[-1][-1]


'''
Coin Change 2
Number of ways you can create change

'''


class Solution:
    def change(self, amount: int, coins: List[int]) -> int:

        dp = [0 for i in range(amount + 1)]
        dp[0] = 1

        for coin in coins:

            for j in range(coin, amount + 1):
                if coin <= j:
                    dp[j] += dp[j - coin]

        return dp[amount]

#  Knapsack perspective

#         dp = [ [ 0 for i in range(amount + 1) ] for _ in range(len(coins) + 1) ]

#         for i in range(len(dp)):
#             dp[i][0] = 1

#         for i in range(1, len(dp)):
#             coin = coins[i - 1]

#             for j in range(1, len(dp[0])):

#                     if coin <= j:
#                         dp[i][j] = dp[i-1][j] + dp[i][j - coin]
#                     else:
#                         dp[i][j] = dp[i-1][j]


#         return dp[-1][-1]

'''
divide set into two subsets of equal sums

V.V.V important question for understanding memoization.

'''


class Solution:
    def canPartition(self, nums: List[int]) -> bool:

        if sum(nums) % 2 != 0:
            return False

        return self.recurse(0, 0, nums, sum(nums), {})

    def recurse(self, index, crs, nums, total, memo):

        checkMemoString = str(index) + str(crs)
        if checkMemoString in memo:
            return memo[checkMemoString]

        if crs * 2 == total:
            return True

        if index >= len(nums) or crs > total / 2:
            return False

        canPartition = self.recurse(index + 1, crs + nums[index], nums, total, memo) or self.recurse(index + 1, crs,
                                                                                                     nums, total, memo)

        memo[str(index) + str(crs)] = canPartition

        return canPartition

'''
minimum palindrome partition

https://www.algoexpert.io/questions/Palindrome%20Partitioning%20Min%20Cuts
'''


def palindromePartitioningMinCuts(string):
    dp = [[False for sj in string] for si in string]
    for i in range(len(string)):
        dp[i][i] = True
    for length in range(2, len(dp) + 1):
        for i in range(0, len(string) - length + 1):
            j = i + length - 1
            if length == 2:
                dp[i][j] = string[i] == string[j]
            else:
                dp[i][j] = (string[i] == string[j]) and dp[i + 1][j - 1]

    cuts = [float("inf") for _ in string]
    for i in range(len(string)):
        if dp[0][i]:
            cuts[i] = 0
        else:
            cuts[i] = cuts[i - 1] + 1
            for j in range(1, i):
                if dp[j][i] and cuts[j - 1] + 1 < cuts[i]:
                    cuts[i] = cuts[j - 1] + 1

    return cuts[-1]

'''
0 - 1 Knapsack

brute force : Time O ( 2 ^ N ) Space: O(1)

'''

def knapsackProblem(items, capacity):
    matrix = [[0 for i in range(capacity + 1)] for _ in range(len(items) + 1)]

    for i in range(1, len(matrix)):
        weight, value = items[i - 1][1], items[i - 1][0]
        for j in range(1, len(matrix[0])):
            if j - weight < 0:
                matrix[i][j] = matrix[i - 1][j]
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i - 1][j - weight] + value)

    print(matrix)

    return matrix[-1][-1]

'''
0 - 1 knapsack and return the exact weights selected

'''


def knapsackProblem(items, capacity):
    matrix = [[0 for i in range(capacity + 1)] for _ in range(len(items) + 1)]

    for i in range(1, len(matrix)):
        weight, value = items[i - 1][1], items[i - 1][0]
        for j in range(1, len(matrix[0])):
            if j - weight < 0:
                matrix[i][j] = matrix[i - 1][j]
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i - 1][j - weight] + value)

    print(matrix)

    return [matrix[-1][-1], getItemsAdded(matrix, items)]


def getItemsAdded(matrix, items):
    result = []

    i = len(matrix) - 1
    c = len(matrix[0]) - 1

    while i > 0:
        if matrix[i][c] == matrix[i - 1][c]:
            i -= 1
        else:
            result.append(i - 1)
            c -= items[i - 1][1]
            i -= 1
        if c == 0:
            break

    return result


'''
VVVV important

Distinct subsequences

Recursive solution : Time O( M X N ) Space O( M ) 
Iterative solution : Time O(M X N ) Space O ( M X N ) 
Optimized iterative soltion : Time O(M X N) Space O(N)

https://leetcode.com/problems/distinct-subsequences/

'''

class Solution:
    def numDistinct(self, s: str, t: str) -> int:

        S, T = len(s), len(t)
        memo = {}

        def helper(i, j):
            if i == S or j == T:
                return 1 if j == T else 0
            if (i, j) in memo:
                return memo[i, j]
            ans = 0
            if s[i] == t[j]:
                ans = helper(i + 1, j) + helper(i + 1, j + 1)
            else:
                ans = helper(i + 1, j)

            memo[i, j] = ans
            return ans

        return helper(0, 0)


class Solution:
    def numDistinct(self, s: str, t: str) -> int:

        S, T = len(s), len(t)

        dp = [[0 for i in range(T + 1)] for j in range(S + 1)]
        for i in range(S + 1):
            dp[i][T] = 1

        for i in range(S - 1, -1, -1):
            for j in range(T - 1, -1, -1):

                dp[i][j] = dp[i + 1][j]
                if s[i] == t[j]:
                    dp[i][j] += dp[i + 1][j + 1]

        return dp[0][0]


class Solution:
    def numDistinct(self, s: str, t: str) -> int:

        S, T = len(s), len(t)

        dp = [0 for i in range(T)]

        for i in range(S - 1, -1, -1):
            prev = 1
            for j in range(T - 1, -1, -1):

                oldDJ = dp[j]
                if s[i] == t[j]:
                    dp[j] += prev

                prev = oldDJ

        return dp[0]

'''
Distinct Subsequence II

TLE solution 
Optimzed solution with DP. 
adding one unique char doubles the number of distinct subsequnces
'''

class Solution:
    seen = set()
    def helper(index, comb, n, S):
        if index >= n:
            if comb and str(comb) not in seen:
                seen.add(str(comb))
            return

        if S[index] not in seen:
            seen.add(str([S[index]]))

        if comb and str(comb) not in seen:
            seen.add(str(comb))
            return

        for i in range(index + 1, n + 1):
            helper(i, comb + [S[index]], n, S)
            helper(i, comb, n, S)

        return


    helper(0, [], len(S), S)
    return len(seen)


class Solution:
    def distinctSubseqII(self, S: str) -> int:
        dp = [1]
        last = {}

        for i in range(len(S)):
            char = S[i]
            dp.append(dp[-1] * 2)
            if char in last:
                dp[-1] -= dp[last[char]]
            last[char] = i
        print(dp)
        return (dp[-1] - 1) % (10 ** 9 + 7)
