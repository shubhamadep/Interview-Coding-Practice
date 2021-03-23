'''
Longest substring without repeating characters
https://leetcode.com/problems/longest-substring-without-repeating-characters/

'''
class Solution:
    def lengthOfLongestSubstring(self, string: str) -> int:

        if not string:
            return 0

        max_length = 0
        start = 0
        seen = collections.defaultdict(int)

        for end in range(len(string)):
            char = string[end]
            if char in seen and seen[char] >= start:
                start = seen[char] + 1
            seen[char] = end
            max_length = max(max_length, end - start + 1)

        return max_length

'''
https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/submissions/

'''
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        n = len(s)
        ans = 0
        freq = Counter(s)
        max_nums = len(freq)
        for num in range(1, max_nums + 1):
            counter = defaultdict(int)
            left = 0
            for right in range(n):
                counter[s[right]] += 1

                while len(counter) > num:
                    counter[s[left]] -= 1
                    if counter[s[left]] == 0:
                        del counter[s[left]]
                    left += 1
                for key in counter:
                    if counter[key] >= k:
                        flag = 1
                    else:
                        flag = 0
                        break
                if flag == 1:
                    ans = max(ans, right - left + 1)
        return ans
