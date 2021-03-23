'''

relative sort

https://leetcode.com/problems/relative-sort-array/

'''

class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:

        counts = collections.Counter(arr1)
        result = []

        for num in arr2:
            result.extend([num for _ in range(counts[num])])
            del counts[num]

        for num in sorted(list(counts.keys())):
            result.extend([num for _ in range(counts[num])])
            del counts[num]

        return result

'''
Largest subarray of sum 0
https://www.geeksforgeeks.org/find-the-largest-subarray-with-0-sum/

Brute force algorithm: O(n^2) time and O(1) space
Optimal solution uses O(n) time and O(n) space ( hashmap )

Proof:

sum till i -- a[0] + a[1] + a[2] + ... a[i]
sum till j -- a[0] + a[1] + a[2] + ... a[j]

is sum till i == sum til j and j > i. That means the sum is seen before and the subarray from i -> j has sum 0. 
ie. length of subarray is j - i + 1
 
'''

class Solution:
    def largestSubArrayWithSumZero(self, array):
        previousSums = {}
        sum = 0
        maxLenght = 0

        for i in range(len(array)):
            sum += array[i]

            if array[i] == 0 and maxLenght == 0:
                maxLenght = 1

            if sum == 0:
                maxLenght = i + 1

            if sum in previousSums:
                maxLenght = max( maxLenght, i - previousSums[sum] + 1)
            else:
                previousSums[sum] = i

        return maxLenght

'''
Number of subarrays with sum k 
https://leetcode.com/problems/subarray-sum-equals-k/

similar problem to above problem. 
'''


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:

        previousSum = {0: 1}
        currSum = 0
        count = 0

        for num in nums:
            currSum += num

            if currSum - k in previousSum:
                count += previousSum[currSum - k]

            previousSum[currSum] = previousSum.get(currSum, 0) + 1

        return count

'''
https://www.geeksforgeeks.org/find-a-pair-swapping-which-makes-sum-of-two-arrays-same/

sumA - a + b = sumB - b + a
2a - 2b  = sumA - sumB
a - b  = (sumA - sumB) / 2

now use 2 sum to find that. You might have to sort the arrays first.
Time:O(n + m) Space: O(nlogn + mlogm) -- sorting

'''

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

'''
Array Pair Sum Divisibility Problem 

'''
import collections
class Solution:
    def arrayDiv(self, nums, k):
        result = []
        seen = collections.defaultdict()
        for i in range(len(nums)):
            num = nums[i]
            if seen[k-num%num]:
                # do something towards anser
                pass
            seen[k - num % num] = i
        return result


'''
minimum swaps needed to make two strings same

Solution: https://leetcode.com/problems/minimum-swaps-to-make-strings-equal/discuss/419498/Python-O(n)-defaultdict-beats-100

'''


class Solution:
    def minimumSwap(self, s1: str, s2: str) -> int:

        counts = collections.Counter(list(s1))
        for c in s2:
            counts[c] += 1

        for count in counts:
            if counts[count] % 2 == 1:
                return -1

        difference = collections.defaultdict(int)
        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                difference[(c1, c2)] += 1

        answer = 0
        for diff in difference:
            pairs = difference[diff] // 2
            remaining = difference[diff] % 2
            answer += pairs + remaining

        return answer

'''
Minimum number of moves to make to all numbers equal 

https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/

Time: O(nlogn)
Space: O(1)

'''


class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        nums.sort()
        count = 0
        median = nums[len(nums) // 2]

        for num in nums:
            count += abs(median - num)

        return count

'''
VVV important
number of subarrays with sum equal to k 

https://leetcode.com/problems/subarray-sum-equals-k/

'''


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:

        previousSum = {0: 1}
        currSum = 0
        count = 0

        for num in nums:
            currSum += num

            if currSum - k in previousSum:
                count += previousSum[currSum - k]

            previousSum[currSum] = previousSum.get(currSum, 0) + 1

        return count

'''
Revise smallest subarray of sum k from this folder. VVV important

'''

'''
https://leetcode.com/problems/minimum-window-substring/

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
                if targetCount[s[right]] > 0:
                    count += 1
                targetCount[s[right]] -= 1

            while count == len(t):
                if right - left + 1 < minWindowLength:
                    minWindowLength = right - left + 1
                    result = s[left: right + 1]

                if s[left] in targetCount:
                    targetCount[s[left]] += 1
                    if targetCount[s[left]] > 0:
                        count -= 1

                left += 1

        return result
