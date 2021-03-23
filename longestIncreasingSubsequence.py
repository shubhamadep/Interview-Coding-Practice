'''
This method is very important in many problems

The below solution in O(n*n) solution, another O(nlogn) solution is available.

'''


def longestIncreasingSubsequence(nums):
    lengths = [1 for _ in range(len(nums))]
    sequences = [None for _ in range(len(nums))]

    for i in range(1, len(nums)):

        num = nums[i]
        for j in range(i):
            # find number with longest string smallest to the current number.
            if nums[j] < num:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    sequences[i] = j

    return backtrack(sequences, lengths, nums)


def backtrack(sequences, lengths, nums):
    mx, index = 1, 0
    for i in range(1, len(lengths)):
        if lengths[i] > mx:
            mx = lengths[i]
            index = i

    result = []
    while index != None:
        result.append(nums[index])
        index = sequences[index]

    return result[::-1]


