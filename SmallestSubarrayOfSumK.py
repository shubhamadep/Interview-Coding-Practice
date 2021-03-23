'''
Must look at this question

https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/

P[i1] + P[i2] .... P[i3] -- profit at each index / sum at each index

We need to find y such that P[y] - p[i] >= k ; i is the largest

brute force is O(n^3)
basic solution is O(n^2)

You use a double queue: front and back and keep P[i+3] > P[i+1] > P[i] property

because if we have an x such that x < i+3 and P[i+3] > P[x]. So, if P[y] - P[x] >= K then P[y] - P[i+3] >= 3 because i+3 is greater and
we want to maximize i+3

explanation : https://www.youtube.com/watch?v=_JDpJXzTGbs&ab_channel=TechLifeWithShivank

'''


class Solution:
    def shortestSubarray(self, A: List[int], K: int) -> int:

        profits = [0]
        for num in A:
            profits.append(profits[-1] + num)

        ans = float('inf')
        q = collections.deque()
        for i in range(len(profits)):

            py = profits[i]
            while q and py - profits[q[0]] >= K:
                ans = min(ans, i - q[0])
                q.popleft()

            while q and py - profits[q[-1]] <= 0:
                q.pop()

            q.append(i)

        return ans if ans != float('inf') else -1