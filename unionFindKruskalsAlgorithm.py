'''

Union function: to group elements
find function: to find to include in groups

'''

class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.components = n
        self.size = [1] * n

    def find(self, p):
        root = p
        while root != self.parent[root]:
            root = self.parent[root]

        while p != root:
            p, self.parent[p] = self.parent[p], root

        return root

    def union(self, p, q):
        parent_p = self.find(p)
        parent_q = self.find(q)

        if parent_p == parent_q:
            return

        if self.size[parent_p] > self.size[parent_q]:
            self.parent[parent_q] = parent_p
            self.size[parent_p] += self.size[parent_q]
        else:
            self.parent[parent_p] = parent_q
            self.size[parent_q] += self.size[parent_p]

        self.components -= 1

    def isConnected(self, p, q):
        return self.find(p) == self.find(q)


class Solution:
    def minimumCost(self, N, connections):
        connections.sort(key=lambda x: x[2])
        UF = UnionFind(N)
        minCost = 0

        for c1, c2, cost in connections:
            if not UF.isConnected(c1 - 1, c2 - 1):
                UF.union(c1 - 1, c2 - 1)
                minCost += cost

        if all(UF.isConnected(i, i+1) for i in range(N-1)):
            return minCost
        else:
            return -1
