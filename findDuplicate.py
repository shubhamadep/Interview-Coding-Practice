'''
visit the index and convert it into negative .
When visited again just check if its less than 0. If so return the index. 
'''

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:

        for i in range(len(nums)):
            index = abs(nums[i])
            if nums[index] < 0:
                return index
            nums[index] = - nums[index]
