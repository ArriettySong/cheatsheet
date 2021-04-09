"""
题目：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array
"""
from typing import List

class Solution:
    # 遍历查找
    def findMin(self,nums:List[int])->int:
        if not nums:
            return

        res = nums[0]
        for i in range(1,len(nums)):
            if nums[i]>=nums[i-1]:
                i += 1
            else:
                res = nums[i]
        return res

    # 二分查找
    def findMinByPivot(self,nums:List[int])->int:
        if not nums:
            return

        low,high = 0,len(nums)-1
        while low<high:
            pivot = low + (high-low) // 2   # // 取整除 - 返回商的整数部分（向下取整）
            if nums[pivot] < nums[high]:
                high = pivot
            else :
                low = pivot + 1
        return nums[low]

nums = [3,4,5,1,2]
# nums = [1,2,3,4,5]
solution = Solution()
print(solution.findMin(nums))
print(solution.findMinByPivot(nums))