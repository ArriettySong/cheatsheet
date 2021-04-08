"""
题目: https://leetcode-cn.com/problems/volume-of-histogram-lcci/

"""

""": 动态规划法
题解：https://leetcode-cn.com/problems/volume-of-histogram-lcci/solution/zhi-fang-tu-de-shui-liang-by-leetcode-so-7rla/
对于下标 i，水能到达的【最大高度】等于下标 i 两边的最大高度的最小值。下标 i 处能接的水的量等于下标 i 处的水能到达的最大高度减去 height[i]。
复杂度分析:
时间复杂度：O(n)，其中 n 是数组 height 的长度。计算数组 leftMax 和 rightMax 的元素值各需要遍历数组 height 一次，计算能接的水的总量还需要遍历一次。
空间复杂度：O(n)，其中 n 是数组 height 的长度。需要创建两个长度为 n 的数组 leftMax 和 rightMax。
"""

from typing import List
class Solution_DynamicProgramming:
    def trap(self,height:List[int])-> int:
        if not height:
            return 0

        n = len(height) # list长度
        left_max = [height[0]] + [0] * (n-1)    # 从左边扫描，记录每个i位置左边最大的值（包含自己）
        right_max = [0] * (n-1) + [height[n-1]] # 从左边扫描，记录每个i位置右边最大的值（包含自己）

        for i in range(1,n):
            left_max[i] = max(left_max[i-1],height[i])

        for i in range(n-2,-1,-1):
            right_max[i] = max(right_max[i+1],height[i])

        return sum( min(left_max[i],right_max[i])-height[i] for i in range(n))

""":单调栈
除了计算并存储每个位置两边的最大高度以外，也可以用单调栈计算能接的水的总量。
维护一个单调栈，单调栈存储的是下标，满足从栈底到栈顶的下标对应的数组 height 中的元素递减。
复杂度分析：
时间复杂度：O(n)，其中 n 是数组 height 的长度。从 0 到 n−1 的每个下标最多只会入栈和出栈各一次。
空间复杂度：O(n)，其中 n 是数组 height 的长度。空间复杂度主要取决于栈空间，栈的大小不会超过 n。
"""

class Solution_MonotonicStack:
    def trap(self,height:List[int])-> int:
        if not height:
            return 0
        res = 0
        stack = list()
        n = len(height)

        for index, value in enumerate(height):
            while stack and value > height[stack[-1]]:
                top = stack.pop()
                if not stack:
                    break
                left = stack[-1]
                currWidth = index - left - 1
                currHeight = min(height[left], height[index]) - height[top]
                res += currWidth * currHeight
            stack.append(index)
        return res


""":双指针
双指针，分别从左右出发，谁小谁动，直至将小的一侧的所有比较大的所有点都过完，最终指针汇聚在最高点。
复杂度分析：
时间复杂度：O(n)，其中 n 是数组 height 的长度。两个指针的移动总次数不超过 n。
空间复杂度：O(1)。只需要使用常数的额外空间。
"""

class Solution_DoublePointer:
    def trap(self,height:List[int])-> int:
        if not height:
            return 0

        left,right = 0, len(height)-1
        leftmax, rightmax = 0, 0
        res = 0

        while left < right :
            leftmax = max(leftmax,height[left])
            rightmax = max(rightmax,height[right])
            if height[left] < height[right]:
                res += leftmax-height[left]
                left += 1
            else:
                res += rightmax-height[right]
                right -= 1
        return res


height = [0,1,0,2,1,0,1,3,2,1,2,1]
# height = [5,1,2,1]
solution1 = Solution_DynamicProgramming()
print(solution1.trap(height))

solution2 = Solution_MonotonicStack()
print(solution2.trap(height))

solution3 = Solution_DoublePointer()
print(solution3.trap(height))