'''
https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/submissions/

If you have n-1 connections and n-1 nodes and you have to change all the routes to point to one of the nodes.

Imagine this question like a tree.

create a undirected graph with connections with booleans to get directions.  -- The main destination will become the root.
Then use BFS to traverse.

'''


class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        visited = [False for _ in range(n)]
        visited[0] = True
        pathChanges = 0

        adjList = self.createAdjList(connections, n)
        q = collections.deque([0])

        while q:
            node = q.popleft()
            for destination, direction in adjList[node]:
                if visited[destination]:
                    continue
                if not direction:
                    pathChanges += 1
                visited[destination] = True
                q.append(destination)

        return pathChanges

    def createAdjList(self, connections, n):

        adjList = {}

        for i in range(n):
            adjList[i] = []

        for connection in connections:
            adjList[connection[0]].append((connection[1], False))
            adjList[connection[1]].append((connection[0], True))

        return adjList

'''
Course scheduling

https://leetcode.com/problems/course-schedule/
'''


class Solution(object):
    def canFinish(self, numCourses, prerequisites):

        # edge cases
        if not prerequisites:
            return numCourses

        # adjecency list
        adj, indegree = collections.defaultdict(list), {}
        count = 0

        for i in range(numCourses):
            indegree[i] = 0

        # indegree, and adj matrices of courses
        for i in range(len(prerequisites)):
            c, d = prerequisites[i]
            adj[d].append(c)
            indegree[c] = indegree.get(c) + 1

        print(indegree, adj)
        stack = []
        for k in indegree:
            if indegree[k] == 0:
                stack.append(k)

        while stack:
            count += 1
            top = stack.pop()
            print(top)
            for neighbor in adj[top]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    stack.append(neighbor)

        return count == numCourses

'''
Minimum number of nodes to reach all nodes

https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/
'''


class Solution(object):
    def canFinish(self, numCourses, prerequisites):

        # edge cases
        if not prerequisites:
            return numCourses

        # adjecency list
        adj, indegree = collections.defaultdict(list), {}
        count = 0

        for i in range(numCourses):
            indegree[i] = 0

        # indegree, and adj matrices of courses
        for i in range(len(prerequisites)):
            c, d = prerequisites[i]
            adj[d].append(c)
            indegree[c] = indegree.get(c) + 1

        print(indegree, adj)
        stack = []
        for k in indegree:
            if indegree[k] == 0:
                stack.append(k)

        while stack:
            count += 1
            top = stack.pop()
            print(top)
            for neighbor in adj[top]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    stack.append(neighbor)

        return count == numCourses

'''
Number of provinces
https://leetcode.com/problems/number-of-provinces/

'''

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:

        def dfs(index, visited):
            for j in range(len(isConnected[0])):
                if isConnected[index][j] == 1 and not visited[j]:
                    visited[j] = 1
                    dfs(j, visited)

        provinces = 0
        visited = [False for _ in range(len(isConnected))]

        for i in range(len(isConnected)):
            if not visited[i]:
                dfs(i, visited)
                provinces += 1

        return provinces

'''
Critical connections
https://leetcode.com/problems/critical-connections-in-a-network/submissions/

'''


class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        def generateGraph(n, connections):
            self.rank = {}
            self.graph = collections.defaultdict(list)
            self.connection_edges = {}

            for i in range(n):
                self.rank[i] = None

            for edges in connections:
                u, v = edges[0], edges[1]

                self.graph[u].append(v)
                self.graph[v].append(u)
                self.connection_edges[(min(u, v), max(u, v))] = 1

        def dfs(node, currentRank):

            if self.rank[node]:
                return self.rank[node]

            self.rank[node] = currentRank
            minRank = currentRank + 1

            for neighbor in self.graph[node]:
                if self.rank[neighbor] and self.rank[neighbor] == currentRank - 1:
                    continue
                recursiveRank = dfs(neighbor, currentRank + 1)
                if recursiveRank <= currentRank:
                    del self.connection_edges[(min(node, neighbor), max(node, neighbor))]

                minRank = min(minRank, recursiveRank)

            return minRank

        generateGraph(n, connections)
        dfs(0, 0)

        result = []
        for u, v in self.connection_edges:
            result.append([u, v])

        return result
