'''

In this problem its importnat to first get the count of the string .
now for chars with occcurance less than k, divide the string into parts .
compare lengths of part on left and part of the right recursively .

return the max

'''

class Solution:
    def longestSubstring(self, s: str, k: int) -> int:

        if not s:
            return 0

        counter = collections.Counter(s)

        left = 0
        while counter[s[left]] >= k and left < len(s):
            left += 1
            if left >= len(s):
                return left

        left = self.longestSubstring(s[0:left], k)

        '''
        This is just an optimization to increment left until valid.

        '''
        right = left + 1
        while right < len(s) and counter[s[right]] < k:
            right += 1

        right = self.longestSubstring(s[right:], k) if right < len(s) else 0

        return max(left, right)

        
