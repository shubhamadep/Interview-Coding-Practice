'''
Number of burgers with no waste of ingredients
https://leetcode.com/problems/number-of-burgers-with-no-waste-of-ingredients/

So, the equation would be,
4x + 2y = tomatoSlices which means, total number of tomatoSlices required for making x jumbo burgers and y small burgers should be EXACTLY tomatoSlices (if possible)

Second equation would be,
x + y = cheeseSlices which means, total number of cheeseSlices required for making x jumbo burgers and y small burgers should be EXACTLY cheeseSlices (if possible)

'''
class Solution:
    def numOfBurgers(self, t: int, c: int) -> List[int]:
        '''
        solving equations

        4x + 2y = T
        x + y = c

        4x + 2(c - x) = T
        x = ( T - 2c ) // 2
        y = c - x

        '''

        x = (t - 2 * c) // 2
        y = c - x
        if x >= 0 and y >= 0 and x + y == c and 4 * x + 2 * y == t:
            return [x, y]
        return []