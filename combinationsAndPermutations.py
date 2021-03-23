'''
For question where you need to find all the combinations / permutations in the list to add up to something .
You need to use DFS.
And create combinations as you go deep in the tree. if found add it to result.

'''

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res = []
        self.candidates = candidates
        self.dfs(target, 0, [])

        return self.res

    def dfs(self, t, start, comb):
        if t == 0:
            self.res.append(comb)
            return
        if t < 0:
            return

        for i in range(start, len(self.candidates)):
            c = self.candidates[i]
            self.dfs(t - c, i, [ c ] + comb)

        return

'''
For finding all the combinations in the list
This uses back tracking

Time complexity is K . CKN ( CKN time required to build the combinations and K is length of the output. SO time required to append the output. )
'''
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:

        result = []

        def gen_comb(n, k, start, cur_comb):

            if k == len(cur_comb):
                # base case, also known as stop condition

                result.append( cur_comb[::] )
                return

            else:
                # general case:

                # solve in DFS
                for i in range(start, n+1):

                    cur_comb.append( i )

                    gen_comb(n, k, i+1, cur_comb)

                    cur_comb.pop()

                return
        # ----------------------------------------------

        gen_comb( n, k, start=1, cur_comb=[] )
        return result

'''
Create all permutations in the string / list

Time complexity : N . N . N!
'''

class Solution:
    def helper(self, arr, perm, perms, length):
        # to check if arr is empty and if given input array was empty
        if not len(arr) and len(perm):
            perms.append(perm)
        else:
            for i in range(len(arr)):
                newarr = arr[:i] + arr[i + 1:]
                newperm = perm + [arr[i]]
                self.helper(newarr, newperm, perms, length)

    def getPermutations(self, array):
        permutations = []
        self.helper(array, [], permutations, len(array))
        return permutations
