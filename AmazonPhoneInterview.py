'''
Target Sum pro
find a + b + c = z

ie. a + b = z - c

https://leetcode.com/discuss/interview-question/763964/Amazon-or-Phone-or-Target-Sum-Pro
Time: O(n^2)
Space: O(n)

'''

'''
Group IDs by pair

https://leetcode.com/discuss/interview-question/690707/Amazon-or-Phone-or-Group-Product-Id-pairs-into-Categories

1. create adjacency list T:O(n) S:O(n)
2. visited set S: O(n)
3. DFS for keys in adjList T: O(n)
 
'''

class Solution:
    def TwoSum(self, array, target):
        if not array:
            return False

        prevSeen = set()
        for num in array:
            if num - target not in prevSeen:
                prevSeen.add(num-target)
            else:
                return [num, num-target]