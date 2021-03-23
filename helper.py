def solution(self):
    def maximum_points(self, nums):
        def helper(self, nums, points, nums_count, seen):
            if len(seen) == len(nums_count):
                return 0

            for num in nums:
                temp_points = 0
                if num not in seen:
                    temp_points += nums_count[num] * num
                    seen.add(num)
                    if num - 1 in nums_count:
                        temp_points += nums_count[num - 1] * (num - 1)
                        seen.add(num - 1)
                    if num + 1 in nums_count:
                        temp_points += nums_count[num + 1] * (num + 1)
                        seen.add(num + 1)

                    self.maximum_points = max(self.maximum_points,
                                              helper(nums.remove(num), points + temp_points, nums_count, seen))

        self.maximum_points = float('inf')
        nums_count = {}
        for num in nums:
            if num in nums:
                nums_count[num] += 1
            else:
                nums_count[num] = 1

        self.helper(nums, 0, nums_count, set())
        return self.maximum_points