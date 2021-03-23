'''
This structure can be used for longest palindrome subsequence type questions

basic logic to remember here is

if outer elements are true, then check if internal elements were palindrome or not.

if s[j] == s[i]:
    dp[i][j] = dp[i+1][j-1] + 2
else:
    dp[i][j] = max( dp[i+1][j], dp[i][j-1] )


'''
 dp = [[ 0 for _ in range(len(s))] for _ in range(len(s))]

        for i in range(len(s)):
            dp[i][i] = 1

        for i in range(1, len(s)):
            if s[i-1] == s[i]:
                dp[i-1][i] = 2
            else:
                dp[i-1][i] = 1

        for gap in range(2, len(s)):
            for i in range(len(s) - gap):
                j = i + gap

'''
For finding longest palindrome substring

every element has odd / even palindrome lenght. Get max keep track of i, j

'''
    def longestPalindrome(self, s: str) -> str:
        if len(s) < 2:
            return s

        longest = [0, 1]

        for i in range(1, len(s)):
            odd = self.palindrome(s, i-1, i+1)
            even = self.palindrome(s, i-1, i)
            longest = max(longest, odd, even, key=lambda x: x[1] - x[0])

        return s[longest[0]: longest[1]]

    def palindrome(self, s, i, j):

        while i >= 0 and j < len(s) :

            if s[i] == s[j]:
                i -= 1
                j += 1
            else:
                break

        return [i+1, j]


'''
Palindromic Substrings

'''

class Solution:
    def function(self,s):
        dp = [[ 0 for _ in range(len(s))] for _ in range(len(s))]
        count = 0

        for i in range(len(s)):
            dp[i][i] = 1
            count += 1

        for i in range(1, len(s)):
            if s[i-1] == s[i]:
                dp[i-1][i] = 1
                count += 1

        for gap in range(2, len(s)):
            for i in range(len(s) - gap):
                j = i + gap
                if s[i] == s[j] and dp[i+1][j-1] == 1:
                    dp[i][j] = 1
                    count += 1


        return count

'''
longest palindrome subsequence

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

        return dp[0][-1]


