'''
To practice:
https://www.geeksforgeeks.org/longest-palindromic-subsequence-dp-12/
https://www.geeksforgeeks.org/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum/
merging k sorted arrays
Given a matrix, rotate by 90 degree anti-clockwise (inplace)
https://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion-and-without-stack/
deep copy linked list
sp

'''

'''
Permutations in strings

https://leetcode.com/problems/permutation-in-string/

Time : O(l1 + l2)
Space: O(1)

Can use sliding window approach to optimize. Explaining this with the brute force permutation approach is enough.
'''


class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if not s1:
            return False

        if len(s1) > len(s2):
            return False

        s1Counts = collections.Counter(s1)

        for i in range(len(s2) - len(s1) + 1):

            s2Counts = collections.Counter(s2[i:i + len(s1)])

            if self.matches(s1Counts, s2Counts):
                return True

        return False

    def matches(self, x, y):

        for char in x:
            if x[char] != y[char]:
                return False

        return True

'''
Anagrams

sorting : O(NLOGN) space: O(1)
Hash map: O(N) space: O(N)
'''

'''
Good question 

https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/
Find longest palindrome subsequence and then subtract it from length of string

'''


class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1

        for gap in range(2, n + 1):
            for i in range(n - gap + 1):
                j = i + gap - 1
                if s[i] == s[j] and gap == 2:
                    dp[i][j] = 2
                elif s[i] == s[j]:
                    dp[i][j] = 2 + dp[i + 1][j - 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

        return len(s) - dp[0][-1]

'''
Longest distinct character subarray

https://leetcode.com/problems/longest-substring-without-repeating-characters/
'''


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0

        st, end = 0, 0
        seen = {}
        n = len(s)
        mx = 0

        for end in range(n):
            if s[end] not in seen or seen[s[end]] < st:
                mx = max(end - st + 1, mx)
                seen[s[end]] = end
            else:
                st = seen[s[end]] + 1
                seen[s[end]] = end

        return mx

'''
longest common prefix in list of words

find prefix between two words, and pass it. Until we iterate through all the strings. 

'''


class Solution:

    def longestUntil(self, str1, str2):
        common = ""
        i = 0
        j = 0
        n1 = len(str1)
        n2 = len(str2)
        while (i < n1 and j < n2):
            if (str1[i] == str2[j]):
                common += str1[i]
            else:
                break
            i += 1
            j += 1
        return common

    def longestCommonPrefix(self, strs: List[str]) -> str:
        n = len(strs)
        if (n == 0):
            return ""
        prefix = strs[0]
        for i in range(1, n):
            prefix = self.longestUntil(prefix, strs[i])

        return prefix

'''
Split string into maximum number of unique substrings

Tip: WHen you see maximum number of minumum number of in a string, try thinking about recursion / DP. recursion is possible
if the bounds given are on the lower end. In this problem the max number of characters is 16. So its possible to 
exhaust the recursion. 

Time : O(N! . N . N) 
Space : O(N)
'''


class Solution:
    def maxUniqueSplit(self, s: str) -> int:
        self.result = 0

        def helper(index, seen):
            if index == len(s):
                self.result = max(self.result, len(seen))

            for end in range(index + 1, len(s) + 1):
                curr = s[index:end]
                if curr not in seen:
                    seen.add(curr)
                    helper(end, seen)
                    seen.remove(curr)
                    print(seen)

        helper(0, set())
        return self.result

'''
remove continous duplicates

'''


class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        if not s:
            return ""

        stack = [["-1", -1]]
        count = 0

        for letter in s:

            if stack[-1][0] != -1 and stack[-1][0] == letter:
                stack.append([letter, stack[-1][1] + 1])
            else:
                stack.append([letter, 1])

            if stack[-1][1] == k:
                for _ in range(k):
                    stack.pop()
                count = 0

        if not stack[1:]:
            return ""

        result = ""
        for char in stack[1:]:
            result += char[0]

        return result

'''
remove subsequent duplicates
'''

class Solution:
    def smallestSubsequence(self, text: str) -> str:
        countMap = collections.defaultdict(int)
        stack = []
        selected = set()

        for c in text:
            countMap[c] += 1

        for c in text:
            countMap[c] -= 1
            if c not in selected:
                while stack and countMap[stack[-1]] > 0 and stack[-1] > c:
                    selected.remove(stack.pop())

                stack.append(c)
                selected.add(c)

        return len(stack)