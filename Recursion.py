'''
word break

inital solution: O( n^n ) space O(n)
with memoization: O( n^2 ) space O(n)

'''


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        memo = {}

        def helper(s, wd, st):

            if st == len(s):
                return True

            if st in memo:
                return memo[st]

            for end in range(st + 1, len(s) + 1):
                val = helper(s, wd, end)
                if s[st:end] in wd and val:
                    return True
                memo[end] = val

            return False

        wd = set(wordDict)
        return helper(s, wd, 0)



