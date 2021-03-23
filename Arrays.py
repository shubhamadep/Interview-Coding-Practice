'''
Kadane's algorithm

1. If all numbers are positive: You can use two pointers to get the maximum subarray sum
2. If array contains negative numbers: You need to keep track of the max sum and change the continuous sum accordingly

Types of sums:
1. Normal array: Use Kadane's
2. Given circular array, where last element points to the start of the array:

Type 1:

'''


def kadanesAlgorithm(array):
    cs = array[0]
    ms = array[0]

    for i in range(1, len(array)):
        cs = max(cs + array[i], array[i])
        ms = max(cs, ms)

    return ms

'''
Type 2

{1,2,3,7,5}

Given circular array, where last element points to the start of the array:

Keep track of the min continuos just like we track max sum in Kadene's. Logic: When you subtract min subarray sum from Sum(array) you will 
get the circular sum which can be compared to the max sub array sum calculated by Kadene's.

Keep in mind the edge case: All numbers are negative. Then you need to check the min sum and Sum(array) are equal. and return max sub array sum.  

'''


class Solution:

    def maxSubarraySumCircular(self, A: List[int]) -> int:
        currentMax = A[0]
        maxSoFar = A[0]
        currentMin = A[0]
        minSoFar = A[0]

        for num in A[1:]:
            currentMax = max(num, currentMax + num)
            maxSoFar = max(currentMax, maxSoFar)
            currentMin = min(num, currentMin + num)
            minSoFar = min(currentMin, minSoFar)

        if sum(A) == minSoFar:
            return maxSoFar

        return max(maxSoFar, sum(A) - minSoFar)

'''
Merge two sorted arrays ( NOT LINKED LIST )

normal solution is to use extra space and two pointers.
To optimize you can merge in constant space. O(1) space

[1, 2, 3, 0, 0, 0]
       p1       p
[2, 5, 6]
       p2
       
compare p2 and p1 and start filling at p and updating pointers. 

'''


class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        # two get pointers for nums1 and nums2
        p1 = m - 1
        p2 = n - 1
        # set pointer for nums1
        p = m + n - 1

        # while there are still elements to compare
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] = nums1[p1]
                p1 -= 1
            p -= 1

        # add missing elements from nums2
        nums1[:p2 + 1] = nums2[:p2 + 1]

'''

[9,6,4,2,3,5,7,0,1]

Missing number in an array
0 - N
len(array) -> N-1

https://leetcode.com/problems/missing-number/

sum of the range : n(n+1) // 2
sum of range - sum of array

'''


def missingNumber(self, nums: List[int]) -> int:
    missing = len(nums)
    for i in range(missing):
        missing ^= i ^ nums[i]

    return missing

'''

First missing smallest number
[1, 7, 3, 4]
[0, 1, 2, 3]
Answer: 2 ( because index 2 should have 2 )

https://leetcode.com/problems/first-missing-positive/

'''

def firstMissingPositive(nums):
    for i in range(len(nums)):

        correctPos = nums[i] - 1
        while 1 <= nums[i] <= len(nums) and nums[i] != nums[correctPos]:
            nums[i], nums[correctPos] = nums[correctPos], nums[i]
            correctPos = nums[i] - 1

    for i in range(len(nums)):
        if i + 1 != nums[i]:
            return i + 1

    return len(nums) + 1


'''
Sort array by parity

O(N) time 
O(1) space

https://leetcode.com/problems/sort-array-by-parity/

'''


class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:

        p, q = 0, len(A) - 1
        while p < q:
            if A[p] % 2 > A[q] % 2:
                A[p], A[q] = A[q], A[p]

            if A[p] % 2 == 0:
                p += 1
            if A[q] % 2 == 1:
                q -= 1

        return A

'''
https://practice.geeksforgeeks.org/problems/-rearrange-array-alternately/0/   

Import concept: You can store two numbers in one index by using arr[i] += (arr[max_idx] % max_elem) * max_elem

'''


def rearrange(arr, n):
    # Initialize index of first minimum
    # and first maximum element
    max_idx = n - 1
    min_idx = 0

    # Store maximum element of array
    max_elem = arr[n - 1] + 1

    # Traverse array elements
    for i in range(0, n):

        # At even index : we have to put maximum element
        if i % 2 == 0:
            arr[i] += (arr[max_idx] % max_elem) * max_elem
            max_idx -= 1

        # At odd index : we have to put minimum element
        else:
            arr[i] += (arr[min_idx] % max_elem) * max_elem
            min_idx += 1

    # array elements back to it's original form
    for i in range(0, n):
        arr[i] = arr[i] / max_elem

'''
count number of inversions

https://www.geeksforgeeks.org/counting-inversions/

Time : O(nlogn)
Space : O(n)

important to note that number of inversion when merging two subarrays is 
if A[j] < A[i] : number of inversions = mid - i

[1, 2] [3, 0]

'''
class Solution:
    def countInversion(self, array):
        return self.mergeSort(array, [0 for _ in range(len(array))], 0, len(array) - 1)

    def mergeSort(self, array, sortedArray, left, right):

        inversions = 0

        if left < right:
            mid = left + ( right - left ) // 2
            inversions += self.mergeSort(array, sortedArray, left, mid)
            inversions += self.mergeSort(array, sortedArray, mid+1, right)
            inversions += self.merge(array, sortedArray, left, mid, right)

        return inversions

    def merge(self, array, sortedArray, left, mid, right):
        i, j = left, mid + 1
        k = left
        inversions = 0

        while i <= mid and j < right:
            if array[i] <= array[j]:
                sortedArray[k] = array[i]
                i += 1
            else:
                sortedArray[k] = array[j]
                inversions += mid - i
                j += 1

            k += 1

        while i <= mid:
            sortedArray[k] = array[i]
            i += 1
            k += 1

        while j < right:
            sortedArray[k] = array[j]
            j += 1
            k += 1

        return inversions

'''
sort colors / sort arrays of 0s, 1s, 2s

Time: O(n) Space: O(1)

'''


class Solution:
    def sortColors(self, nums: List[int]) -> None:

        def swap(x, y):
            nums[x], nums[y] = nums[y], nums[x]

        p, curr = 0, 0
        q = len(nums) - 1

        while curr <= q:
            if nums[curr] == 0:
                swap(curr, p)
                p += 1
                curr += 1
            elif nums[curr] == 2:
                swap(curr, q)
                q -= 1

            else:
                curr += 1

'''
equilibrium point
https://leetcode.com/problems/find-pivot-index/

Time: O(N) Space: O(1)
'''


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:

        Sum = sum(nums)
        cs = 0

        for i in range(len(nums)):
            cs += nums[i]
            print(cs, Sum - cs)
            if cs - nums[i] == Sum - cs:
                return i

        return -1

'''
Partition equal subset sum

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
leader in an array

sleaders-in-an-array/0

'''

'''
minimum number of platforms

https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station-set-2-map-based-approach/

'''

'''
reverse array in groups of k
Time O(N) , Space: O(1)

'''
class Solution:
    def reverse(self, array, k):
        def swap(self, x, y):
            array[x], array[y] = array[y], array[x]

        i = 0
        while i < len(array):
            left = i
            right = min(i+k, len(array)-1)

            while left < right:
                swap(left, right)
                left += 1
                right -= 1
            i += k
        return array


'''

Kth largest : min heap 
Kth smallest: max heap 

Heap Data structure
Time: O(NlogN)
Space: O(K)

Optimized: Quick Select
Time: O(N)
worst: O(n**2) - when you choose extreme pivots
space: O(1)

'''


class KthLargest:

    def __init__(self, k: int, nums: List[int]):

        self.heap = []
        self.k = k
        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, val)

        return self.heap[0]

'''
Trapping rain water
https://leetcode.com/problems/trapping-rain-water/solution/

'''
class Solution:
    def trappedWater(self, array):
        leftMax, rightMax, left, right = 0, 0, 0, len(array) - 1
        ans = 0

        while left < right:

            # if array[left] < array[right] : water on left will depend on leftMax
            if array[left] < array[right]:
                if array[left] >= leftMax:
                    leftMax = array[left]
                else:
                    ans += leftMax - array[left]
                left += 1
            else:
                if array[right] >= rightMax:
                    rightMax = array[right]
                else:
                    ans += rightMax - array[right]
                right -= 1

        return ans

'''
Chocolate distribution problem
https://www.geeksforgeeks.org/chocolate-distribution-problem/

Time: O(NlogN) Space: O(1)

'''
class Solution:
    def distribute(self, array, k):
        if not array:
            return 0

        array.sort()
        ans = float('inf')
        for i in range(len(array) - k):
            j = i + k - 1
            ans = min(ans, array[j] - array[i])

        return ans

'''

https://www.geeksforgeeks.org/convert-array-into-zig-zag-fashion/

Arr[] = {4, 3, 7, 8, 6, 2, 1}
Output: 3 7 4 8 2 6 1
Explanation: 3 < 7 > 4 < 8 > 2 < 6 > 1

sort = {1, 2, 3, 4, 6, 7, 8}
       {1, 3, 2, 6, 4, 8, 7}

'''


def zigZag(arr, n):
    # Flag true indicates relation "<" is expected,
    # else ">" is expected. The first expected relation
    # is "<"
    flag = True
    for i in range(n - 1):
        # "<" relation expected
        if flag is True:
            # If we have a situation like A > B > C,
            # we get A > B < C
            # by swapping B and C
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                # ">" relation expected
        else:
            # If we have a situation like A < B < C,
            # we get A < C > B
            # by swapping B and C
            if arr[i] < arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
        flag = bool(1 - flag)
    print(arr)


# Driver program
arr = [4, 3, 7, 8, 6, 2, 1]
n = len(arr)
zigZag(arr, n)

'''
Spiral traverse matrix

'''


def spiralTraverse(array):
    result = []

    startRow, endRow = 0, len(array) - 1
    startCol, endCol = 0, len(array[0]) - 1

    while startRow <= endRow and startCol <= endCol:

        # top row
        for col in range(startCol, endCol + 1):
            result.append(array[startCol][col])

        # right col
        for row in range(startRow + 1, endRow + 1):
            result.append(array[row][endCol])

        # bottom row
        for col in reversed(range(startCol, endCol)):
            # if only one row in middle, you already calculated the values
            if startRow == endRow:
                return result
            result.append(array[endRow][col])

        # left row
        for row in reversed(range(startRow + 1, endRow)):
            if startCol == endCol:
                return result
            result.append(array[row][startCol])

        startRow += 1
        endRow -= 1
        startCol += 1
        endCol -= 1

    return result

'''
Candy problem / Min rewards problem 

Questions to ask: All positives ? 

https://www.algoexpert.io/questions/Min%20Rewards
https://leetcode.com/problems/candy/

'''


def minRewards(scores):
    rewards = [1 for _ in range(len(scores))]

    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            rewards[i] = max(rewards[i], rewards[i - 1] + 1)

    for i in reversed(range(len(scores) - 1)):
        if scores[i] > scores[i + 1]:
            rewards[i] = max(rewards[i], rewards[i + 1] + 1)

    print(rewards)
    return sum(rewards)

'''
Largest number formed in array of numbers

Using merge sort
Time: O(NlogN)
Space: O(N)
'''

#Leetcode solution
class LargerNumKey(str):
    def __lt__(x, y):
        return x + y > y + x


class Solution:
    def largestNumber(self, nums):
        largest_num = ''.join(sorted(map(str, nums), key=LargerNumKey))
        return '0' if largest_num[0] == '0' else largest_num

#should use this in interview.

class Solution:
    def largestNumber(self, nums: List[int]) -> str:

        def mergeSort(nums):

            if len(nums) <= 1:
                return nums

            mid = len(nums) // 2
            left = mergeSort(nums[:mid])
            right = mergeSort(nums[mid:])

            return merge(left, right)

        def merge(left, right):
            sortedArray = []
            i, j = 0, 0

            while i < len(left) and j < len(right):
                if int(str(left[i]) + str(right[j])) >= int(str(right[j]) + str(left[i])):
                    sortedArray.append(str(left[i]))
                    i += 1
                else:
                    sortedArray.append(str(right[j]))
                    j += 1

            while i < len(left):
                sortedArray.append(str(left[i]))
                i += 1

            while j < len(right):
                sortedArray.append(str(right[j]))
                j += 1

            return sortedArray

        sortedNums = mergeSort(nums)
        #if we have a string as "00" we need to return "0" so we int it and str again
        return str(int("".join(map(str, sortedNums))))


'''
Find all anagrams in a string
https://leetcode.com/problems/find-all-anagrams-in-a-string/

'''


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:

        ns, np = len(s), len(p)
        if ns < np:
            return []

        p_count = Counter(p)
        s_count = Counter()

        output = []
        # sliding window on the string s
        for i in range(ns):
            # add one more letter
            # on the right side of the window
            s_count[s[i]] += 1
            # remove one letter
            # from the left side of the window
            if i >= np:
                if s_count[s[i - np]] == 1:
                    del s_count[s[i - np]]
                else:
                    s_count[s[i - np]] -= 1
            # compare array in the sliding window
            # with the reference array
            if p_count == s_count:
                output.append(i - np + 1)

        return output

'''
Median of two sorted arrays
https://leetcode.com/problems/median-of-two-sorted-arrays/

'''


class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        if len(nums2) < len(nums1):
            return self.findMedianSortedArrays(nums2, nums1)

        x = len(nums1)
        y = len(nums2)
        low = 0
        high = x

        while low <= high:
            partx = math.floor((low + high) / 2)
            party = math.floor(((x + y + 1) / 2) - partx)

            maxleftx = float('-inf') if partx == 0 else nums1[partx - 1]
            minrightx = float('inf') if partx == x else nums1[partx]

            maxlefty = float('-inf') if party == 0 else nums2[party - 1]
            minrighty = float('inf') if party == y else nums2[party]

            if maxleftx <= minrighty and maxlefty <= minrightx:
                if (x + y) % 2 == 0:
                    return (max(maxleftx, maxlefty) + min(minrighty, minrightx)) / 2
                else:
                    return max(maxleftx, maxlefty)
            elif maxleftx > minrighty:
                high = partx - 1
            else:
                low = partx + 1

