'''
Gas station
https://leetcode.com/problems/gas-station/

'''

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:

        n = len(gas)
        total_tank, current_tank = 0, 0
        starting_position = 0

        for i in range(n):
            total_tank += gas[i] - cost[i]
            current_tank += gas[i] - cost[i]
            if current_tank < 0:
                starting_position = i + 1
                current_tank = 0

        return starting_position if total_tank >= 0 else -1