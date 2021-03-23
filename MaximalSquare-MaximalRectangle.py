class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix: return 0

        dp = [[0 for _ in range(len(matrix[0]) + 1)] for _ in range(len(matrix) + 1)]
        largest = 0

        for row in range(1, len(dp)):
            for col in range(1, len(dp[0])):
                if matrix[row - 1][col - 1] == "1":
                    dp[row][col] = min(dp[row - 1][col - 1], dp[row - 1][col], dp[row][col - 1]) + 1
                    largest = max(largest, dp[row][col])

        return largest * largest