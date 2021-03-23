'''
Use collections.deque() to implement use queue in python

'''

class MyStack:

    def __init__(self):
        self.q1 = collections.deque()


    def push(self, x: int) -> None:
        self.q1.append(x)
        count = len(self.q1)
        while count > 1:
            self.q1.append(self.q1.popleft())
            count -= 1


    def pop(self) -> int:
        return self.q1.popleft()

    def top(self) -> int:
        return self.q1[0]


    def empty(self) -> bool:
        return not self.q1



# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

'''
maximum in subarray of size K
https://leetcode.com/submissions/detail/386900471/

'''


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:

        q = collections.deque()
        result = []

        for i in range(len(nums)):

            num = nums[i]
            # pop elements which are smaller than this element
            while q and num > nums[q[-1]]:
                q.pop()

            # dequeue elements which are outside range
            while q and i - q[0] >= k:
                q.popleft()

            q.append(i)
            result.append(nums[q[0]])

        return result[k - 1:]

