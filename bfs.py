'''
Rotten oranges

'''

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:

        queue = collections.deque()

        hasRoute = False

        for i in range(len(grid)):
            for j in range(len(grid[0])):

                if grid[i][j] == 2:

                    queue.append((i, j))
                    hasRoute = True

                elif grid[i][j] == 1:
                    hasRoute = True

        if not hasRoute:
            return 0

        time = self.bfs(grid, queue)

        return time if self.checkGrid(grid) else -1

    def checkGrid(self, grid):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    return False

        return True

    def bfs(self, grid, queue):

        time = -1
        directions = [
            (+1, 0), (0, +1), (-1, 0), (0, -1)
        ]
        rows = len(grid)
        cols = len(grid[0])
        while queue:

            time += 1
            children = []

            while queue:
                children.append(queue.pop())

            for child in children:
                i, j = child
                grid[i][j] = 0


                for direction in directions:
                    r, c = i + direction[0], j + direction[1]
                    if r < rows and r >= 0 and c < cols and c >=0:
                        if grid[r][c] == 1:
                            grid[r][c] = 2
                            queue.append((r, c))



        return time

'''
Number of distinct islands

'''


class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:

        def dfs(row, col, direction):
            if check(row, col):
                return

            if (row, col) in seen or not grid[row][col]:
                return

            seen.add((row, col))
            path_signature.append(direction)
            for r, c, d in directions:
                new_r, new_c = row + r, col + c
                if not check(new_r, new_c):
                    dfs(new_r, new_c, d)
            path_signature.append("O")

        def check(row, col):
            return row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0])

        seen = set()
        directions = [(+1, 0, 'D'), (-1, 0, 'U'), (0, +1, 'R'), (0, -1, 'L')]
        unique_islands = set()

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                path_signature = []
                dfs(i, j, "O")
                if path_signature:
                    unique_islands.add(tuple(path_signature))

        return len(unique_islands)