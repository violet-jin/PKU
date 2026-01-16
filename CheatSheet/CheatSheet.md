# 一、动态规划 (Dynamic Programming)
## 1. DP 解题框架
 1. 确定dp数组含义
 2. 确定递推公式
 3. dp数组初始化
 4. 确定遍历顺序
 5. 举例推导验证

### 斐波那契数列
```
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```
# 空间优化版本
```
def fibonacci_optimized(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```
2. 经典DP问题模板
# 01背包问题
```
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # 倒序遍历，防止重复使用
        for w in range(capacity, weights[i]-1, -1):
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    return dp[capacity]
```
# 完全背包问题
```
def knapsack_complete(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # 正序遍历，可以重复使用
        for w in range(weights[i], capacity + 1):
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    return dp[capacity]
```
# 最长公共子序列 (LCS)
```
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```
3. 矩阵DP
# 最小路径和
```
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # 初始化第一行和第一列
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    # 状态转移
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[m-1][n-1]
```
二、深度优先搜索 (DFS)
1. 递归DFS模板
# 基本递归DFS
```
def dfs_recursive(node, visited):
    if node in visited:
        return
    visited.add(node)
    # 处理当前节点
    print(node)
    # 遍历相邻节点
    for neighbor in get_neighbors(node):
        dfs_recursive(neighbor, visited)
```
# 带路径的DFS
```
def dfs_with_path(start, target):
    def dfs(node, path, visited):
        if node == target:
            result.append(path[:])
            return
        visited.add(node)
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()  # 回溯

        visited.remove(node)
    result = []
    dfs(start, [start], set())
    return result
```
2. 迭代DFS模板
# 使用栈的迭代DFS
```
def dfs_iterative(start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            # 处理当前节点
            # 将邻居节点压栈（注意顺序）
            for neighbor in reversed(get_neighbors(node)):
                if neighbor not in visited:
                    stack.append(neighbor)
    return visited
```
3. 常见DFS应用
# 二叉树的最大深度
```
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```
# 二叉树的所有路径
```
def binary_tree_paths(root):
    def dfs(node, path):
        if not node:
            return
        path.append(str(node.val))
        if not node.left and not node.right:
            paths.append("->".join(path))
        else:
            dfs(node.left, path)
            dfs(node.right, path)
        path.pop()
    paths = []
    dfs(root, [])
    return paths
```
三、广度优先搜索 (BFS)
1. 基础BFS模板
```
from collections import deque

def bfs(start, target):
    # 队列用于BFS
    queue = deque([start])
    # 记录访问过的节点
    visited = set([start])
    # 记录步数（如果需要）
    steps = 0

    while queue:
        # 当前层的节点数
        level_size = len(queue)

        for _ in range(level_size):
            node = queue.popleft()

            # 找到目标
            if node == target:
                return steps

            # 遍历邻居
            for neighbor in get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        steps += 1

    return -1  # 未找到
```
# 带路径的BFS
def bfs_with_path(start, target):
    queue = deque([[start]])
    visited = set([start])

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == target:
            return path

        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path[:]
                new_path.append(neighbor)
                queue.append(new_path)

    return []  # 未找到路径
2. 层次遍历模板
# 二叉树层次遍历
def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
3. 双向BFS（优化版）
def bidirectional_bfs(start, target):
    if start == target:
        return 0

    # 两个方向分别的队列
    front_queue, back_queue = deque([start]), deque([target])
    # 两个方向分别的访问记录
    front_visited, back_visited = {start: 0}, {target: 0}

    while front_queue and back_queue:
        # 从较小的一端扩展
        if len(front_queue) <= len(back_queue):
            if expand_queue(front_queue, front_visited, back_visited):
                return front_visited[target] + back_visited[target]
        else:
            if expand_queue(back_queue, back_visited, front_visited):
                return front_visited[target] + back_visited[target]

    return -1

def expand_queue(queue, visited, other_visited):
    # BFS扩展一层
    level_size = len(queue)

    for _ in range(level_size):
        node = queue.popleft()

        for neighbor in get_neighbors(node):
            if neighbor in other_visited:
                return True

            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)

    return False
四、回溯算法 (Backtracking)
1. 回溯通用模板
def backtrack_template(choices):
    result = []

    def backtrack(path, choices):
        # 终止条件
        if meet_condition(path):
            result.append(path[:])  # 复制一份
            return

        for choice in choices:
            # 做出选择
            if is_valid(choice, path):
                path.append(choice)
                # 进入下一层决策树
                backtrack(path, new_choices(choices, choice))
                # 撤销选择
                path.pop()

    backtrack([], choices)
    return result
2. 具体应用示例
# 全排列
def permute(nums):
    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for num in nums:
            if num not in path:  # 剪枝：避免重复
                path.append(num)
                backtrack(path)
                path.pop()

    result = []
    backtrack([])
    return result

# 组合问题
def combine(n, k):
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)  # i+1避免重复
            path.pop()

    result = []
    backtrack(1, [])
    return result

# N皇后问题
def solve_n_queens(n):
    def is_valid(board, row, col):
        # 检查列
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # 检查左上对角线
        i, j = row-1, col-1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # 检查右上对角线
        i, j = row-1, col+1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return

        for col in range(n):
            if is_valid(board, row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'

    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(0)
    return result
3. 剪枝优化
# 组合总和（可重复使用）
def combination_sum(candidates, target):
    def backtrack(start, path, current_sum):
        if current_sum == target:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            # 剪枝：如果当前数字已经超过目标值
            if current_sum + candidates[i] > target:
                continue

            path.append(candidates[i])
            # i而不是i+1，因为可以重复使用
            backtrack(i, path, current_sum + candidates[i])
            path.pop()

    result = []
    candidates.sort()  # 排序以便剪枝
    backtrack(0, [], 0)
    return result
五、二分查找 (Binary Search)
1. 基础二分模板
# 标准二分查找（查找确切值）
def binary_search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # 未找到
2. 变种二分查找
# 查找第一个等于target的位置
def first_equal(nums, target):
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] >= target:
            if nums[mid] == target:
                result = mid
            right = mid - 1
        else:
            left = mid + 1

    return result

# 查找最后一个等于target的位置
def last_equal(nums, target):
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] <= target:
            if nums[mid] == target:
                result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result

# 查找第一个大于等于target的位置
def first_greater_equal(nums, target):
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] >= target:
            result = mid
            right = mid - 1
        else:
            left = mid + 1

    return result

# 查找最后一个小于等于target的位置
def last_less_equal(nums, target):
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] <= target:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result
3. 旋转数组中的二分查找
# 旋转数组中查找最小值
def find_min_in_rotated(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return nums[left]

# 旋转数组中搜索目标值
def search_in_rotated(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # 判断哪一半是有序的
        if nums[left] <= nums[mid]:  # 左半部分有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # 右半部分有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
六、双指针 (Two Pointers)
1. 快慢指针
# 判断链表是否有环
def has_cycle(head):
    if not head or not head.next:
        return False

    slow, fast = head, head.next

    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next

    return True

# 找到链表环的入口
def detect_cycle(head):
    slow = fast = head

    # 第一阶段：判断是否有环
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:  # 有环
            # 第二阶段：找到环入口
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow

    return None

# 找到链表中间节点
def middle_node(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow
2. 左右指针
# 两数之和（有序数组）
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []

# 三数之和
def three_sum(nums):
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i-1]:  # 去重
            continue

        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1

                # 去重
                while left < right and nums[left] == nums[left-1]:
                    left += 1
                while left < right and nums[right] == nums[right+1]:
                    right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result

# 盛最多水的容器
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        h = min(height[left], height[right])
        w = right - left
        max_water = max(max_water, h * w)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water
3. 滑动窗口
# 最小覆盖子串
def min_window(s, t):
    from collections import Counter

    need = Counter(t)
    window = Counter()

    left = right = 0
    valid = 0
    start, length = 0, float('inf')

    while right < len(s):
        # 扩大窗口
        c = s[right]
        right += 1

        if c in need:
            window[c] += 1
            if window[c] == need[c]:
                valid += 1

        # 收缩窗口
        while valid == len(need):
            # 更新最小覆盖子串
            if right - left < length:
                start = left
                length = right - left

            d = s[left]
            left += 1

            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1

    return "" if length == float('inf') else s[start:start+length]

# 无重复字符的最长子串
def length_of_longest_substring(s):
    window = set()
    left = right = 0
    max_len = 0

    while right < len(s):
        if s[right] not in window:
            window.add(s[right])
            right += 1
            max_len = max(max_len, right - left)
        else:
            window.remove(s[left])
            left += 1

    return max_len

# 滑动窗口最大值
def max_sliding_window(nums, k):
    from collections import deque

    if not nums:
        return []

    result = []
    window = deque()

    for i in range(len(nums)):
        # 移除窗口外的元素
        if window and window[0] <= i - k:
            window.popleft()

        # 移除小于当前元素的元素
        while window and nums[window[-1]] < nums[i]:
            window.pop()

        window.append(i)

        # 添加结果
        if i >= k - 1:
            result.append(nums[window[0]])

    return result

3. 常见算法
# 最大公约数
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# 最小公倍数
def lcm(a, b):
    return a * b // gcd(a, b)

# 质数判断
def sieve(limit):
    """埃拉托色尼筛法，返回素数列表"""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
    return primes
一、基础模板
1. 标准实现
class UnionFind:
    def __init__(self, n):
        """初始化n个元素的并查集"""
        self.parent = list(range(n))  # 每个元素的父节点
        self.rank = [0] * n          # 秩（树的高度）
        self.count = n               # 连通分量个数

    def find(self, x):
        """查找x的根节点（路径压缩优化）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        """合并x和y所在的集合"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # 已经在同一集合

        # 按秩合并（小的树接到大的树上）
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1  # 连通分量减少
        return True

    def connected(self, x, y):
        """判断x和y是否连通"""
        return self.find(x) == self.find(y)

    def get_count(self):
        """返回连通分量个数"""
        return self.count
2. 简洁版（适合竞赛）
# 简化版本，不按秩合并
class UnionFindSimple:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x
            return True
        return False

    def connected(self, x, y):
        return self.find(x) == self.find(y)
3. 带权并查集（维护额外信息）
class WeightedUnionFind:
    def __init__(self, n):
        """带权并查集，可以维护节点到根节点的距离"""
        self.parent = list(range(n))
        self.weight = [0] * n  # 节点到父节点的权值

    def find(self, x):
        """查找根节点，同时更新权值"""
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] += self.weight[self.parent[x]]  # 更新权值
            self.parent[x] = root
        return self.parent[x]

    def union(self, x, y, w):
        """
        合并x和y，使得weight[x] - weight[y] = w
        w表示x比y大多少
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            # 检查是否矛盾
            return self.weight[x] - self.weight[y] == w

        # 合并，更新权值
        self.parent[root_x] = root_y
        self.weight[root_x] = w + self.weight[y] - self.weight[x]
        return True

    def diff(self, x, y):
        """返回x - y的值"""
        if self.find(x) != self.find(y):
            return None  # 不在同一集合
        return self.weight[x] - self.weight[y]
模板1：标准连通性问题
def solve_with_union_find(n, connections):
    """
    标准连通性问题的解决方案模板
    """
    uf = UnionFind(n)

    # 处理连接
    for u, v in connections:
        uf.union(u, v)

    # 如果需要找出所有连通分量
    components = {}
    for i in range(n):
        root = uf.find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    return list(components.values())
模板2：最小生成树（Kruskal算法）
def kruskal(n, edges):
    """
    Kruskal算法求最小生成树
    edges: [(weight, u, v)]
    """
    # 按权重排序
    edges.sort()
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []

    for weight, u, v in edges:
        if uf.union(u, v):
            mst_weight += weight
            mst_edges.append((u, v, weight))

        # 如果已经找到n-1条边，可以提前退出
        if len(mst_edges) == n - 1:
            break

    if len(mst_edges) != n - 1:
        # 图不连通
        return None

    return mst_weight, mst_edges
模板3：判断图是否有环
def has_cycle(n, edges):
    """判断无向图是否有环"""
    uf = UnionFind(n)

    for u, v in edges:
        if not uf.union(u, v):
            # 如果u和v已经在同一集合，说明有环
            return True

    return False

