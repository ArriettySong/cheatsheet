"""
题目: https://leetcode-cn.com/problems/volume-of-histogram-lcci/

"""

""": 双指针法
题解：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/solution/shan-chu-pai-xu-shu-zu-zhong-de-zhong-fu-yec2/
使用双指针解决本题，遍历数组检查每一个元素是否应该被保留，如果应该被保留，就将其移动到指定位置。
具体地，我们定义两个指针 slow 和fast 分别为慢指针和快指针，其中慢指针表示处理出的数组的长度，快指针表示已经检查过的数组的长度，
即 nums[fast]表示待检查的第一个元素，nums[slow−2] 为上一个应该被保留的元素所移动到的指定位置。

复杂度分析:
时间复杂度：O(n)，其中 n 是数组的长度。我们最多遍历该数组一次。
空间复杂度：O(1)。我们只需要常数的空间存储若干变量。
"""

from typing import List
class Solution_DeleteDuplicates:
    def delete_duplicates(self,nums:List[int])-> int:
        if not nums:
            return 0
        if len(nums)<=2:
            return len(nums),nums

        slow=2

        for fast in range(2,len(nums)):
            if nums[fast] != nums[slow-2]:
                nums[slow] = nums[fast]
                slow += 1
        return slow,nums[0:slow]


nums = [1,1,1,2,2,2,3,3]
solution = Solution_DeleteDuplicates()
print(solution.delete_duplicates(nums))

