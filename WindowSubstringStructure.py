'''

In these types of question you need to store what you need to have in the target in a dictionary.
Then you create a window with two pointer, start both at the beginning or both at the end.

Then shrink the window according to the conditions.

'''

def minWindow(self, s, t):

        if not s or not t:
            return ""

        targetCount = {}
        for char in t:
            if char in targetCount:
                targetCount[char] += 1
            else:
                targetCount[char] = 1

        left, count = 0, 0
        minWindowLength = float("inf")
        result = ""

        for right in range(len(s)):


'''
Minimum window Substring

'''
class Solution(object):
    def minWindow(self, s, t):

        if not s or not t:
            return ""

        targetCount = {}
        for char in t:
            if char in targetCount:
                targetCount[char] += 1
            else:
                targetCount[char] = 1

        left, count = 0, 0
        minWindowLength = float("inf")
        result = ""

        for right in range(len(s)):
            if s[right] in targetCount:
                '''
                only increment count if we havent satisfied the count of characters needed.
                '''
                if targetCount[s[right]] > 0:
                    count += 1
                targetCount[s[right]] -= 1

            '''
            This phase is shrinking the window to find the minimum window. Keep shrinking until our count matches the lenght of the target.
            '''
            while count == len(t):
                if right - left + 1 < minWindowLength:
                    minWindowLength = right - left + 1
                    result = s[left: right+1]

                if s[left] in targetCount:
                    targetCount[s[left]] += 1
                    if targetCount[s[left]] > 0:
                        count -= 1

                left += 1

'''
 Longest Substring with At Most Two Distinct Characters

'''

class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:

        distinctSet = {}
        right, left = 0, 0
        maxLength = float("-inf")

        while right < len(s):

            char = s[left]
            if len(distinctSet) < 3:
                distinctSet[char] = right
                right += 1

            print(distinctSet)
            if len(distinctSet) == 3:

                del_index = min(distinctSet.values())
                del distinctSet[s[del_index]]

                left += 1

            maxLength = max(maxLength, rigth - left)

        return maxLength

'''
Tough Question:

https://www.geeksforgeeks.org/find-a-tour-that-visits-all-stations/

Imagine this question as a queue and deque process / or a sliding window question.
'''

'''
Tough # QUESTION:
Sliding Window Maximum

https://leetcode.com/problems/sliding-window-maximum/

'''


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:

        q = collections.deque()
        result = []

        for i in range(len(nums)):

            num = nums[i]
            #pop elements which are smaller than this element
            while q and num > nums[q[-1]]:
                q.pop()

            #dequeue elements which are outside range
            while q and i - q[0] >= k:
                q.popleft()

            q.append(i)
            result.append(nums[q[0]])

        return result[k-1:]
