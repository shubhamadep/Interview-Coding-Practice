'''
Basic stack: Use list to implement stack

'''

'''
The below sum is building min max stack

'''
# Feel free to add new properties and methods to the class.
class MinMaxStack:
	def __init__(self):
		self.minMaxStack = []
		self.stack = []

    def peek(self):
        return self.stack[len(self.stack) - 1]

    def pop(self):
		self.minMaxStack.pop()
        return self.stack.pop()

    def push(self, number):
        newMinMax = {"min": number, "max": number}
		if self.minMaxStack:
			lastElement = self.minMaxStack[len(self.minMaxStack)-1]
			newMinMax["min"] = min(lastElement["min"], newMinMax["min"])
			newMinMax["max"] = max(lastElement["max"], newMinMax["max"])
		self.minMaxStack.append(newMinMax)
		self.stack.append(number)

    def getMin(self):
        return self.minMaxStack[len(self.minMaxStack) - 1]["min"]

    def getMax(self):
        return self.minMaxStack[len(self.minMaxStack) - 1]["max"]



'''
The below sum is building a queue using 2 stacks

'''
class MyQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []


    def push(self, x: int) -> None:
        self.stack1.append(x)

    def pop(self) -> int:
        self.peek()
        return self.stack2.pop()

    def peek(self) -> int:

        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())

        return self.stack2[-1]


    def empty(self) -> bool:
        return not self.stack1 and not self.stack2

'''
Circular tour.

https://www.geeksforgeeks.org/find-a-tour-that-visits-all-stations/

1. Start inserting each station into queue. if petrol is negative, dequeue from 
   front. and enqueue again..
   
'''

'''
Next Greater Element 1
https://leetcode.com/problems/next-greater-element-i/

'''


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = []
        stack = []
        mapping = {}

        for n in nums2:
            while stack and n > stack[-1]:
                mapping[stack.pop()] = n
            stack.append(n)

        while stack:
            mapping[stack.pop()] = -1

        for n in nums1:
            res.append(mapping[n])

        return res

'''
Next Greater Element 2
https://leetcode.com/problems/next-greater-element-ii/

'''


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack = []
        result = [-1 for i in range(len(nums))]
        n = len(nums)

        for i in reversed(range(2 * len(nums))):
            while stack and nums[i % n] >= nums[stack[-1]]:
                stack.pop()
            result[i % n] = nums[stack[-1]] if stack else -1
            stack.append(i % n)

        return result


'''
Basic Calculator - I

'''


class Solution:
    def calculate(self, s: str) -> int:

        operand, n = 0, 0
        res, num = 0, 0
        stack = []
        sign = 1

        for c in s:

            if c.isdigit():
                num = (num * 10) + int(c)
            elif c in ['+', '-']:
                res += sign * num
                sign = 1 if c == '+' else -1
                num = 0
            elif c == '(':
                stack.append(res)
                stack.append(sign)
                sign, res = 1, 0
            elif c == ')':
                res += sign * num
                res *= stack.pop()
                res += stack.pop()
                num = 0

        return res + num * sign
