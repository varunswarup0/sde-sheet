<ol>
<li><h4>
<a href="https://leetcode.com/problems/sort-colors/">Sort 012</a>
</h4></li>

```
i = 0
j = 0
k = n - 1

WHILE i <= k
    if (nums[i] == 1)
        i++

    else if (nums[i] == 0)
        swap(nums, i, j)
        i++
        j++

    else if (nums[i] == 2)
        swap(nums, i, k)
        k--

```

<h4><li><a href="https://leetcode.com/problems/missing-number/">
Missing Number
</a></li></h4>

```

xor = nums.length

i: 0 -> nums.length - 1
xor = xor ^ i ^ nums[i]

missing number: xor

```

<h4><li><a href="https://www.pepcoding.com/resources/data-structures-and-algorithms-in-java-levelup/bit-manipulation/one-repeating-and-one-missing-official/ojquestion">
Missing Number & Repeating Number
</a></li></h4>

```

xor = 0

i : 0 -> nums.length - 1
xor ^= nums[i]

i : 1 -> nums.length
xor ^= i

<- All the bits that are set in xor1 will be set in either x or y ->
RSB = xor & -xor
Elements with RSB not set: x = 0
Elements with RSB set: y = 0

i: 0 -> nums.length - 1
RSB & nums[i] == 0
? x ^= nums[i]
: y ^= nums[i]

i: 1 -> nums.length
RSB & i == 0
? x ^= i
: y ^= i

i: 0 -> n - 1
if(nums[i] == x)
missing number: y
repeating number: x

    if(nums[i] == y)
        missing number: x
        repeating number: y

```

<h4><li><a href="https://leetcode.com/problems/merge-sorted-array/">
Merge two sorted Arrays without extra space
</a></li></h4>

```
i = n - 1
j = m - 1
k = n + m - 1

WHILE (i >= 0 && j >= 0)
nums1[i] > nums2[j]
? nums1[k--] = nums1[i--]
: nums1[k--] = nums2[j--]

WHILE (j >= 0)
nums1[k--] = nums2[j--]

```

<h4><li><a href="https://leetcode.com/problems/maximum-subarray/">
Maximum SubArray
</a></li></h4>

```
approach: Kadane's algorithm
currentSum = -∞
overallSum = -∞

i: 0 -> n
    currentSum = (currentSum >= 0) ? currentSum + val : val
    overallSum = max(overallSum, currentSum)


maximumSum: overallSum

```

<h4><li><a href="https://www.pepcoding.com/resources/online-java-foundation/stacks-and-queues/merge-overlapping-interval-official/ojquestion">
Merge Overlapping Interval
</a></li></h4>

```

given: intervals[n] = {startTime, endTime}
Stack<Interval> st
Stack<Intervals> result

sort(intervals)
st.push(intervals[0])

i: 1 -> n - 1
    currInterval = st.peek

    intervals[i].startTime <= currInterval.endTime
        ? currInterval.endTime = max(currInterval.endTime, intervals[i].endTime)
        : st.push(intervals[i])


WHILE st.size != 0
    result.push(st.pop)

merged intervals: result
```

<h4><li><a href="https://leetcode.com/problems/linked-list-cycle-ii/">
Detect Cycle
</a></li></h4>

```
approach: Floyd's Tortoise and Hare Algorithm

slow = fast = head

WHILE (fast != null && fast.next != null)
slow = slow.next
fast = fast.next.next

    <- if cycle found ->
    if slow == fast
        break

if (fast == null || fast.next == null)
cycle starting point: null
EXIT

slow = head

WHILE (slow != fast)
slow = slow.next
fast = fast.next

cycle starting point: slow

```

<h4><li><a href="https://leetcode.com/problems/find-the-duplicate-number/">
Find the Duplicate Number
</a></li></h4>

```

slow = nums[0]
fast = nums[nums[0]]

<- run till duplicate is found ->
WHILE (slow != fast)
slow = nums[slow]1
fast = nums[nums[fast]]

fast = 0

WHILE (slow != fast)
slow = nums[slow]
fast = nums[fast]

duplicate number: slow

```

<h4><li><a href="https://leetcode.com/problems/set-matrix-zeroes/">
Set Matrix Zeroes
</a></li></h4>

```

isFirstColZero = false
isFirstRowZero = false

<- check the first column ->
i: 0 -> n - 1
if (matrix[i][0] == 0)
isFirstColZero = true
break

<- check the first row ->
i: 0 -> n - 1
if (matrix[0][i] == 0)
isFirstRowZero = true
break

<- check except the first row and column and mark on first row and column ->
i: 1 i -> n - 1
j: 1 -> m
if (matrix[i][j] == 0)
matrix[i][0] = matrix[0][j] = 0

<- process except the first row and column ->
i = 1 -> n
j: 1 -> m
if (matrix[i][0] == 0 || matrix[0][j] == 0)
matrix[i][j] = 0

<- handle the first column ->
if (isFirstColZero)
i: 0 -> n
matrix[i][0] = 0

<- handle the first row ->
if (isFirstRowZero)
i: 0 -> m
matrix[0][i] = 0

```

<h4><li><a href="https://leetcode.com/problems/pascals-triangle/">
Pascal Triangle
</a></li></h4>

```

List<List<Integer>> triangle

i: 0 -> n
List<Integer> row

    j: 0 -> i

        if (j == 0 || j == i)
            row.add(1)

        else
            upLeft = triangle.get(i - 1).get(j - 1)
            upRight = triangle.get(i - 1).get(j)
            row.add(upLeft + upRight)

    triangle.add(row)

}

pascal triangle: triangle

```

<h4><li><a href="https://leetcode.com/problems/next-permutation/">
Next Permutation
</a></li></h4>

```

pivotIndex = -1
pivot = 0

i: n - 1 -> 0
if (nums[i] < nums[i + 1])
pivotIndex = i
pivot = nums[i]
break

if (pivotIndex == -1)
reverse(nums, 0, n)
EXIT

i: n -> 0
if (nums[i] > pivot)
swap(nums, pivotIndex, i)
break

reverse(nums, pivotIndex + 1, n)

```

<h4><li><a href="https://www.geeksforgeeks.org/counting-inversions/">
Count Inversions in an array
</a></li></h4>

```

mergeAndCount (arr, l, m, r)
{

    left = copyOfRange(arr, l, m + 1)
    right = copyOfRange(arr, m + 1, r + 1)

    i = 0
    j = 0
    k = l
    swaps = 0

    WHILE i < left.length AND j < right.length
        if (left[i] <= right[j])
            arr[k++] = left[i++]

        else
            arr[k++] = right[j++]
            swaps += (m + 1) - (l + i)

    WHILE (i < left.length)
        arr[k++] = left[i++]

    WHILE (j < right.length)
        arr[k++] = right[j++]

    number of inversions: swaps

}

mergeSortAndCount(arr, l, r)
{

    count = 0

    if (l < r)
        m = (l + r) / 2

        count += mergeSortAndCount(arr, l, m)
        count += mergeSortAndCount(arr, m + 1, r)
        count += mergeAndCount(arr, l, m, r)

    total number of inversions:  count

}

```

<h4><li><a href="https://leetcode.com/problems/best-time-to-buy-and-sell-stock/">
Best Time to Buy and Sell Stock
</a></li></h4>

```
minPrice = ∞
maxProfit = 0

i: 0 -> n - 1
    if minPrice > prices[i]
        minPrice=prices[i]
    ELIF maxProfit < prices[i]-minPrice
        maxProfit=prices[i]-minPrice

max profit : maxProfit
```

<h4><li><a href="https://leetcode.com/problems/best-time-to-buy-and-sell-stock/">
Best Time to Buy and Sell Stock
</a></li></h4>

```

minPrice = ∞
maxProfit = 0

i: 0 -> n - 1
if (minPrice > prices[i]) minPrice = prices[i]
else if (maxProfit < prices[i]-minPrice) maxProfit = prices[i] - minPrice

max profit : maxProfit

```

<h4><li><a href="https://leetcode.com/problems/rotate-image/">
Rotate Image
</a></li></h4>

```

<- Transpose ->
i: 0 -> n - 1
    j: i -> n - 1
        swap(matrix[i][j], matrix[j][i])

<- Reverse Rows ->
i: 0 -> n
    j: 0 -> (n/2)
        swap(matrix[i][j], matrix[i][n - j - 1])

```

<h4><li><a href="https://leetcode.com/problems/search-a-2d-matrix/">
Search a 2D Matrix
</a></li></h4>

```

start = 0
end = (n * m) - 1

WHILE start <= end
    mid = (start + end) / 2
    i = mid / m
    j = mid % m

    IF matrix[i][j] < target
        start = mid + 1

    ELIF matrix[i][j] > target
        end = mid - 1

    ELSE
        RETURN true

RETURN false
```

<h4><li><a href="https://leetcode.com/problems/search-a-2d-matrix/">
Search a 2D Matrix
</a></li></h4>

```

IF n < 0
    RETURN 1/x * myPow(1/x, -(n + 1))

IF n == 0
    RETURN 1

IF n == 1
    RETURN x

IF n == 2
    RETURN x * x

IF n % 2 == 0
    RETURN myPow(myPow(x, n/2), 2)

ELSE
    RETURN x * myPow(myPow(x, n/2), 2)

```

<h4><li><a href="https://leetcode.com/problems/majority-element/">
Majority Element I (more than n/2 times)
</a></li></h4>

```

count = 0
candidate = 0

i:0 -> n - 1
if (count == 0)
candidate = nums[i]

    count += (nums[i] == candidate) ? 1 : -1

majority element: candidate

```

<h4><li><a href="https://leetcode.com/problems/majority-element-ii/">
Majority Element II (more than n/3 times)
</a></li></h4>

```

List<Integer> result

firstSum = 0
secondSum = 0
firstMajor = ∞
secondMajor = -∞

i: 0 -> n - 1
if (nums[i] == firstMajor)
firstSum++

    else if (nums[i] == secondMajor)
        secondSum++

    else if (firstSum == 0)
        firstMajor = nums[i]
        firstSum = 1

    else if (secondSum == 0)
        secondMajor = nums[i]
        secondSum = 1

    else
        firstSum--
        secondSum--

firstSum = 0
secondSum = 0

i: 0 -> n - 1
if (nums[i] == firstMajor)
firstSum++
else if (nums[i] == secondMajor)
secondSum++

if (firstSum > n/3)
result.add(firstMajor)

if (secondSum > n/3)
result.add(secondMajor)

RETURN result

```

<h4><li><a href="https://leetcode.com/problems/unique-paths/">
Unique Paths
</a></li></h4>

```

dp[m][n]

i: m - 1 -> 0
j: n - 1 -> 0
if (i == m - 1 AND j == n - 1)
dp[i][j] = 1

        else if (i == m - 1)
            dp[i][j] = dp[i][j + 1]

        else if (j == n - 1)
            dp[i][j] = dp[i + 1][j]

        else
            dp[i][j] = dp[i + 1][j] + dp[i][j + 1]

number of unique paths: dp[0][0]

```

<h4><li><a href="https://leetcode.com/problems/reverse-pairs/">
Reverse Pairs
</a></li></h4>

```

number of reverse pairs: mergesort(nums, 0, nums.length-1)

mergesort (nums, low, high)
if (low >= high)
RETURN 0

    mid = low + (high - low) / 2
    count = mergesort(nums, low, mid) + mergesort(nums, mid+1, high)

    i = low -> mid, j = mid+1 -> high
        if (nums[i] > nums[j] * 2)
            count += mid - i + 1
            j++

        else
            i++

    merge(nums, low, high)
    RETURN count

merge (nums, low, high)
mid = low + (high - low) / 2
arr[high - low + 1]

    i = low
    j = mid + 1
    k = 0

    WHILE (k < arr.length)
        num1 = i > mid ? ∞ : nums[i]
        num2 = j > high ? ∞ : nums[j]

        arr[k++] = num1 <= num2 ? nums[i++] : nums[j++]


    for (p = 0 p < arr.length p++)
        nums[p+low] = arr[p]

```

<h4><li><a href="https://leetcode.com/problems/two-sum/">
Two Sum
</a></li></h4>

```

result[2]

HashMap<Integer, Integer> map
i: 0 -> n - 1
if (map.containsKey(target - nums[i]))
result[1] = i
result[0] = map.get(target - nums[i])
RETURN result

    map.put(nums[i], i)

RETURN result

```

<h4><li><a href="https://leetcode.com/problems/3sum/">
3Sum
</a></li></h4>

```

Arrays.sort(nums)
List<List<Integer>> res

i: 0 -> (n - 3)
if i == 0 || (i > 0 && nums[i] != nums[i-1])
lo = i + 1
hi = n - 1
sum = 0 - nums[i]

        WHILE lo < hi
            if nums[lo] + nums[hi] == sum
                res.add(Arrays.asList(nums[i], nums[lo], nums[hi]))

                WHILE lo < hi && nums[lo] == nums[lo + 1]
                    lo++

                WHILE lo < hi && nums[hi] == nums[hi - 1]
                    hi--

                lo++
                hi--

            else if (nums[lo] + nums[hi] < sum)
                lo++

            else
                hi--

RETURN res

```

<h4><li><a href="https://leetcode.com/problems/4sum/">
Four Sum
</a></li></h4>

```

```

<h4><li><a href="https://leetcode.com/problems/longest-consecutive-sequence/">
Longest Consecutive Sequence
</a></li></h4>

```
res = 0
HashMap<Integer, Integer> map

for (n : num)
    if !map.containsKey(n)
        left = (map.containsKey(n - 1)) ? map.get(n - 1) : 0
        right = (map.containsKey(n + 1)) ? map.get(n + 1) : 0

        length = left + right + 1

        map.put(n, length)
        map.put(n - left, length)
        map.put(n + right, length)

        res = max(res, length)


longest consecutive sequence length: res
```

<h4><li><a href="https://leetcode.com/problems/subarray-sum-equals-k/">
Subarray Sum Equals K
</a></li></h4>

```
sum = 0
result = 0
Map<Integer, Integer> preSum
preSum.put(0, 1)

i: 0 -> n - 1
sum += nums[i]
if (preSum.containsKey(sum - k))
result += preSum.get(sum - k)

    preSum.put(sum, preSum.getOrDefault(sum, 0) + 1)

number of subarrays where sum equals k: result
```

<h4><li><a href="https://www.geeksforgeeks.org/count-number-subarrays-given-xor/">
Count the number of subarrays having a given XOR
</a></li></h4>

```

ans = 0
int[n] xorArr

HashMap<Integer, Integer> mp

xorArr[0] = arr[0]

i: 1 -> n - 1
xorArr[i] = xorArr[i - 1] ^ arr[i]

i: 0 -> n - 1
tmp = m ^ xorArr[i]

    ans = ans + (mp.containsKey(tmp) == false ? 0 : ((long)mp.get(tmp)))

    if (xorArr[i] == m)
        ans++

    if (mp.containsKey(xorArr[i]))
        mp.put(xorArr[i], mp.get(xorArr[i]) + 1)
    else
        mp.put(xorArr[i], 1)

number of subarrays having a given XOR: ans

```

<h4><li><a href="https://leetcode.com/problems/longest-substring-without-repeating-characters/">
Longest Substring Without Repeating Characters
</a></li></h4>

```

HashMap<Character, Integer> map
max = 0
j = 0
i: 0 -> n - 1
if map.containsKey(s[i])
j = max(j, map.get(s[i])+1)

    map.put(s[i], i)
    max = max(max, i - j + 1)

longest substring length: max

```

<h4><li><a href="https://leetcode.com/problems/reverse-linked-list/">
Reverse Linked List
</a></li></h4>

```

newHead = null
WHILE (head != null)
next = head.next
head.next = newHead
newHead = head
head = next

new head: newHead

```

<h4><li><a href="https://leetcode.com/problems/middle-of-the-linked-list/">
Middle of the Linked List
</a></li></h4>

```

slow = head
fast = head

WHILE (fast != null && fast.next != null)
slow = slow.next
fast = fast.next.next

middle: slow

```

<h4><li><a href="https://leetcode.com/problems/remove-nth-node-from-end-of-list/">
Remove Nth Node From End of List
</a></li></h4>

```
start = new ListNode(0)
slow = start
fast = start
slow.next = head

i: 1 -> n + 1
    fast = fast.next

WHILE fast != null
    slow = slow.next
    fast = fast.next

slow.next = slow.next.next
new head: start.next
```

<h4><li><a href="https://leetcode.com/problems/delete-node-in-a-linked-list/">
Delete Node in a Linked List
</a></li></h4>

```

node.val = node.next.val
node.next = node.next.next

```

<h4><li><a href="https://leetcode.com/problems/add-two-numbers/">
FAdd Two Numbers
</a></li></h4>

```

dummy = new ListNode(0)
curr = dummy
carry = 0

WHILE l1 != null OR l2 != null
    x = (l1 != null) ? l1.val : 0
    y = (l2 != null) ? l2.val : 0

    sum = x + y + carry

    carry = sum / 10
    curr.next = new ListNode(sum % 10)

    if (l1 != null) l1 = l1.next
    if (l2 != null) l2 = l2.next

    curr = curr.next

if carry > 0
    curr.next = new ListNode(carry)

head of answer linked list: dummy.next

```

<h4><li><a href="https://leetcode.com/problems/intersection-of-two-linked-lists/">
Intersection of Two Linked Lists
</a></li></h4>

```

approach: iterative
space complexity: O(1)

a = headA
b = headB

WHILE( a != b)
    a = a == null ? headB : a.next
    b = b == null ? headA : b.next

intersection point: a
```

<h4><li><a href="https://leetcode.com/problems/reverse-nodes-in-k-group/">
Reverse Nodes in k-Group
</a></li></h4>

```
if head == null OR k == 1
    return head

dummy = new ListNode(0)
dummy.next = head
cur = dummy
next = dummy
pre = dummy

<- find count of the linked list ->
count = 0

WHILE cur.next != null
    cur = cur.next
    count++


WHILE count >= k
    <- Point to first node ->
    cur = pre.next
    <- Point to second node ->
    next = cur.next

    <- k-1 Operation to reverse k list ->
    i: 1 -> k - 1
        cur.next = next.next
        next.next = pre.next
        pre.next = next
        next = cur.next

    pre = cur
    count = count - k


return dummy.next
```

<h4><li><a href="https://leetcode.com/problems/palindrome-linked-list/">
Palindrome Linked List
</a></li></h4>

```

slow = head

isPalindrome = true
Stack<Integer> stack

WHILE slow != null
stack.push(slow.val)
slow = slow.next

WHILE head != null
i = stack.pop()

    if head.val == i
        isPalindrome = true

    else
        isPalindrome = false
        break

    head = head.next

palindrome: isPalindrome

```

<h4><li><a href="https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/">
Flatten a Multilevel Doubly Linked List
</a></li></h4>

```

if head == null
RETURN head

p = head
WHILE p != null
<- CASE 1: No child ->
if p.child == null
p = p.next
continue

    <- CASE 2: Yes child, find tail ->
    Node temp = p.child
    WHILE temp.next != null
        temp = temp.next

    <- Link tail to p.next  ->
    temp.next = p.next
    if p.next != null
        p.next.prev = temp

    <- Connect p with p.child, and remove p.child ->
    p.next = p.child
    p.child.prev = p
    p.child = null

RETURN head

```

<h4><li><a href="https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/">
Flatten a Multilevel Doubly Linked List
</a></li></h4>

```
if head == null
    RETURN null

size = 1
fast = head
slow = head

<- Calculate length ->
WHILE fast.next != null
    size++
    fast = fast.next

<- Put slow.next at the start ->
i: size - (k % size) -> 2
    slow = slow.next

<- Do the rotation ->
fast.next = head
head = slow.next
slow.next = null

RETURN head
```

<h4><li><a href="https://leetcode.com/problems/copy-list-with-random-pointer/">
Clone a Linked List with random and next pointer
</a></li></h4>

```

HashMap<Node, Node> map
pointer = head

WHILE pointer != null
map.put(pointer, new Node(pointer.val))
pointer = pointer.next

pointer = head

WHILE pointer != null
map.get(pointer).next = map.get(pointer.next)
map.get(pointer).random = map.get(pointer.random)
pointer = pointer.next

head: map.get(head)

```

<h4><li><a href="https://leetcode.com/problems/trapping-rain-water/submissions/">
Trapping Rain Water
</a></li></h4>

```

left = 0
right = n - 1

leftMaxHeight = height[left]
rightMaxHeight = height[right]

area = 0

WHILE left < right
leftMaxHeight = max(leftMaxHeight, height[left])
rightMaxHeight = max(rightMaxHeight, height[right])

    if leftMaxHeight < rightMaxHeight
        area += leftMaxHeight - height[left]
        left++

    else
        area += rightMaxHeight - height[right]
        right--

RETURN area

```

<h4><li><a href="https://leetcode.com/problems/remove-duplicates-from-sorted-array/">
Remove Duplicate from Sorted array
</a></li></h4>

```

count = 0

i: 1 -> n - 1
A[i] == A[i - 1]
? count++
: A[i - count] = A[i]

number of unique elements: n - count

```

<h4><li><a href="https://leetcode.com/problems/max-consecutive-ones/">
Max Consecutive Ones
</a></li></h4>

```

maxHere = 0
max = 0

for (n: nums)
maxHere = (n == 0) ? 0 : maxHere + 1
max = max(max, maxHere)

max number of consecutive ones: max

```

<h4><li><a href="https://www.geeksforgeeks.org/find-maximum-meetings-in-one-room/">
Find maximum meetings in one room
</a></li></h4>

```

given: meetings[] = {startTime, endTime, position}
approach: greedy

ArrayList<Integer> result

timeLimit = 0

<- Sort meeting according to finish time ->
Collections.sort(meetings)

result.add(meetings[0].position)

<- timeLimit to check whether new meeting can be conducted or not ->
timeLimit = meetings[0].endTime

<- Check for all meeting whether it can be selected or not ->
i: 1 -> meetings.size - 1
if meetings[i].startTime > timeLimit

        <- Add selected meeting to arraylist ->
        result.add(al[i].position)

        <- Update time limit ->
        timeLimit = al[i].endTime

maximum number of meetings sequence: result

```

<h4><li><a href="https://www.geeksforgeeks.org/activity-selection-problem-greedy-algo-1/">
Activity Selection Problem
</a></li></h4>

```

given: activity[] = {start, finish}

ArrayList<Activity> res

<- Sort activity according to finish time ->
compare(activity)

i = 0
res.add(activity)

j: 1 -> n - 1
if activity[j].start >= activity[i].finish
res.add(activity)
i = j

selected activities: res

```

<h4><li><a href="https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station/">
Minimum Number of Platforms Required for a Railway/Bus Station
</a></li></h4>

```

Arrays.sort(arr)
Arrays.sort(dep)

<- platNeeded indicates number of platforms needed at a time ->
platNeeded = 1
result = 1
i = 1
j = 0

<- Similar to merge in merge sort to process all events in sorted order
WHILE i < n && j < n

    <- If next event in sorted order is arrival increment count of platforms needed ->
    if arr[i] <= dep[j]
        platNeeded++
        i++


    <- Else decrement count of platforms needed ->
    else
        platNeeded--
        j++

    result = max(result, platNeeded)

minimum number of platforms: result

```

<h4><li><a href="https://leetcode.com/problems/maximum-profit-in-job-scheduling/">
Maximum Profit in Job Scheduling
</a></li></h4>

[**38.3. **]()

```

given: jobs[] = {startTime, endTime, profit}

n = startTime.length
int[n][3] jobs

<- Sort meeting according to end time ->
sort(jobs, endTime)

<- <EndTime, Profit> ->
TreeMap <Integer, Integer> dp

dp.put(0, 0)

<- floorEntry: RETURNs a greatest key <= to the given key ->

for job : jobs
cur = dp.floorEntry(job.startTime).getValue + job.profit

    if cur > dp.lastEntry.getValue
        dp.put(job.endTime, cur)

maximum profit: dp.lastEntry.getValue

```

<h4><li><a href="https://www.geeksforgeeks.org/fractional-knapsack-problem/">
Fractional Knapsack Problem
</a></li></h4>

```

given: items[] = {wt, val, ind, cost = val/wt }
approach: greedy

<- Sort items by cost ->
sort(items, cost)

double totalValue = 0d

for i : items

    curWt = i.wt
    curVal = i.val

    if capacity - curWt >= 0
        <- Can be picked whole ->
        capacity -= curWt
        totalValue += curVal

    else
        <- Can't be picked whole ->
        fraction = capacity / curWt
        capacity -= (curWt * fraction))
        totalValue += (curVal * fraction)
        break

max profit: totalValue

```

<h4><li><a href="https://leetcode.com/problems/coin-change/">
Coin Change
</a></li></h4>

```

int dp[amount+1]

i: 1 -> amount
min = Integer.MAX_VALUE

    for coin: coins
        if i - coin >=0 AND dp[i - coin] != -1
            min = min(min, dp[i-coin])

    <- If currentAmount can not be reached, set dp[i] = -1 ->
    dp[i] = (min == Integer.MAX_VALUE) ? -1 : 1 + min

minimum number of coins: dp[amount]

```

<h4><li><a href="https://leetcode.com/problems/subsets/">
Subsets
</a></li></h4>

```
subsets (nums)
sort(nums)

    List<List<Integer>> list
    backtrack(list, new ArrayList, nums, 0)

    RETURN list

backtrack (list, tempList, nums, start)
list.add(tempList)

    i: start -> nums.length - 1
        tempList.add(nums[i])
        backtrack(list, tempList, nums, i + 1)
        tempList.remove(tempList.size - 1)

```

<h4><li><a href="https://leetcode.com/problems/subsets-ii/">
Subsets II (contains duplicates)
</a></li></h4>

```

subsetsWithDup (nums)
List<List<Integer>> list
sort(nums)
backtrack(list, new ArrayList<>(), nums, 0)
RETURN list

private void backtrack (list, tempList, nums, start)
list.add(tempList)

    i: start -> nums.length - 1
        <- Skip duplicates ->
        IF i > start AND nums[i] == nums[i-1]
            CONTINUE

        tempList.add(nums[i])
        backtrack(list, tempList, nums, i + 1)
        tempList.remove(tempList.size() - 1)

```

<h4><li><a href="https://leetcode.com/problems/find-the-duplicate-number/">
Find the Duplicate Number
</a></li></h4>

[**41.3 Combination Sum**](https://leetcode.com/problems/combination-sum/)

```

combinationSum (nums, target)
List<List<Integer>> list
backtrack(list, new ArrayList, nums, target, 0)
RETURN list

backtrack (list, tempList, nums, remain, start)
if remain < 0
RETURN

    ELIF remain == 0
        list.add(tempList)

    else
        i: start -> nums.length - 1
            tempList.add(nums[i])
            <- We can reuse same elements hence i and not i + 1 ->
            backtrack(list, tempList, nums, remain - nums[i], i)
            tempList.remove(tempList.size() - 1)

```

<h4><li><a href="https://leetcode.com/problems/find-the-duplicate-number/">
Find the Duplicate Number
</a></li></h4>

[**41.4 Combination Sum II**](https://leetcode.com/problems/combination-sum-ii/)

```

combinationSum2 (nums, target)
    List<List<Integer>> list
    sort(nums)
    backtrack(list, new ArrayList, nums, target, 0)
    RETURN list

backtrack (list, tempList, nums, remain, start)
    if remain < 0
        RETURN

    ELIF remain == 0
        list.add(tempList)

    else
        i: start -> nums.length - 1
            <- skip duplicates ->
            if i > start AND nums[i] == nums[i-1]
                continue

            tempList.add(nums[i])
            backtrack(list, tempList, nums, remain - nums[i], i + 1)
            tempList.remove(tempList.size() - 1)

```

<h4><li><a href="https://leetcode.com/problems/palindrome-partitioning/">
Palindrome Partitioning
</a></li></h4>

```

partition (s)
    List<List<String>> list
    backtrack(list, new ArrayList, s, 0)
    RETURN list

backtrack(list, tempList, s, start){
    if start == s.length
        list.add(tempList)

    else
        i: start -> s.length - 1
            if isPalindrome(s, start, i)
                tempList.add(s.substring(start, i + 1))
                backtrack(list, tempList, s, i + 1)
                tempList.remove(tempList.size - 1)

```

<h4><li><a href="https://leetcode.com/problems/permutation-sequence/">
K-th Permutation Sequence
</a></li></h4>

```

StringBuilder sb
List<Integer> num
fact = 1

i: 1 i -> n
fact \*= i
num.add(i)

l = k - 1

i: 0 -> n -1
fact /= n - i
index = l / fact
sb.append(num.remove(index))
l -= index \* fact

k-th permutation sequence: sb.toString

```

<h4><li><a href="https://leetcode.com/problems/permutations/">
Permutations (no duplicates
</a></li></h4>

```

permute (nums)
List<List<Integer>> list
backtrack(list, new ArrayList, nums)
RETURN list

backtrack (list, tempList, nums)
if tempList.size == nums.length
list.add(tempList)

else
i: 0 -> nums.length - 1
<- If element already exists, skip ->
if tempList.contains(nums[i])
continue

        tempList.add(nums[i])
        backtrack(list, tempList, nums)
        tempList.remove(tempList.size - 1)

```

<h4><li><a href="https://leetcode.com/problems/n-queens/">
N-Queens
</a></li></h4>

```

approach: bit manipulation

char[][] board
columnBit = 0
normalDiagonal = 0
reverseDiagonal = 0
List<List<String>> res

operateOnBoard (operation, row, col)
board[row][col] = operation == "insert" ? 'Q' : '.'
columnBit ^= 1 << col
normalDiagonal ^= 1 << row + col
reverseDiagonal ^= 1 << row - col + board.length - 1

checkSquare (row, col)

    if columnBit & (1 << col) != 0
        RETURN false

    if normalDiagonal & (1 << (row + col)) != 0
        RETURN false

    if reverseDiagonal & (1 << (row - col + board.length - 1)) != 0
        RETURN false

    RETURN true

solveNQueens (n)
    board char[n][n]

    for row : board
        fill(row, '.')

    solve(0)
    RETURN res

solve (row)
    if row == board.length
        List<String> path

        i: 0 -> board.length - 1
            path.add(new String(board[i]))

        res.add(path)



    col: 0 -> board.length - 1
        if checkSquare(row, col)
            operateOnBoard("insert", row, col)
            solve(row + 1)
            operateOnBoard("remove", row, col)

```

<h4><li><a href="https://leetcode.com/problems/sudoku-solver/">
Sudoku Solver
</a></li></h4>

```

approach: bit manipulation
given: arr int[9][9]
rows int[9]
cols int[9]
grid int[3][3]

operateOnSudoku (operation, i, j, num)
arr[i][j] = (operation == "insert") ? num : 0
rows[i] ^= (1 << num)
cols[j] ^= (1 << num)
grid[i / 3][j / 3] ^= (1 << num)

checkBox (i, j, num)
if rows[i] & (1 << num) != 0
RETURN false

    if cols[j] & (1 << num) != 0
        RETURN false

    if grid[i / 3][j / 3] & (1 << num) != 0
        RETURN false

    RETURN true

solveSudoku (i, j)
if i == arr.length
display arr

    if arr[i][j] != 0
        solveSudoku(j < 8 ? i : i + 1, j < 8 ? j + 1 : 0)

    else
        num: 1 -> 9
            if checkBox(i, j, num)
                operateOnSudoku("insert", i, j, num)
                solveSudoku(j < 8 ? i : i + 1, j < 8 ? j + 1 : 0)
                operateOnSudoku("remove", i, j, num)

main()
i: 0 -> 8
j: 0 -> 8
rows[i] |= (1 << digit)
cols[j] |= (1 << digit)
grid[i / 3][j / 3] |= (1 << digit)

    solveSudoku(0, 0)

```

<h4><li><a href="https://www.geeksforgeeks.org/m-coloring-problem-backtracking-5/">
m Coloring Problem
</a></li></h4>

```

given: ArrayList<Node> nodes = { color = 1, HashSet<Integer> edges }
n: number of edges
m: number of colors

canPaint (nodes, n, m)

    boolean[n + 1] visited
    maxColors = 1

    sv: 1 -> n
        if visited[sv] == true
            continue

        visited[sv] = true
        Queue<Integer> q
        q.add(sv)

        WHILE q.size != 0

            int top = q.remove

            for it: nodes[top].edges
                <- adjacent node color is same, increase it by 1 ->
                if(nodes[top].color == nodes[it].color)
                    nodes[it].color += 1

                maxColors = max(maxColors,
                max(nodes[top].color, nodes[it].color))

                <- number of colors used shoots m, RETURN 0 ->
                if maxColors > m
                    RETURN 0

                <- adjacent node is not visited, mark it visited and push it in queue ->
                if visited[it] == false
                    visited[it] = true
                    q.push(it)

    RETURN maxColors

```

<h4><li><a href="https://www.geeksforgeeks.org/rat-in-a-maze-backtracking-2/">
Rat in a Maze
</a></li></h4>

```

isSafe(maze, x, y)
<- if (x, y outside maze) RETURN false ->
RETURN (x >= 0 && x < N && y >= 0 && y < N && maze[x][y] == 1)

solveMaze (maze)
sol int[N][n]

    if NOT solveMazeUtil(maze, 0, 0, sol)
        RETURN false

    printSolution(sol)
    RETURN true

solveMazeUtil(maze, x, y, sol)
<- If (x, y is goal) RETURN true ->
if x == N - 1 AND y == N - 1 AND maze[x][y] == 1
sol[x][y] = 1
RETURN true

    <- Check if maze[x][y] is valid ->
    if isSafe(maze, x, y)
        <- Check if the current block is already part of solution path ->
        if sol[x][y] == 1
            break

        <- Mark x, y as part of solution path ->
        sol[x][y] = 1

        <- Move +x direction ->
        if solveMazeUtil(maze, x + 1, y, sol)
            RETURN true

        <- Move +y direction ->
        if solveMazeUtil(maze, x, y + 1, sol)
            RETURN true

        <- Move -x direction ->
        if solveMazeUtil(maze, x - 1, y, sol)
            RETURN true

        <- Move -y direction ->
        if solveMazeUtil(maze, x, y - 1, sol)
            RETURN true

        <- Unmark x, y->
        sol[x][y] = 0


    RETURN false

```

<h4><li><a href="https://leetcode.com/problems/word-break/">
Word Break
</a></li></h4>

```

given: String s, List<String> dict
approach: dp

boolean f[s.length + 1]

f[0] = true

i: 1 -> s.length
j: 0 -> i - 1
if f[j] AND dict.contains(s.substring(j, i))
f[i] = true
break

is word break possible: f[s.length()]

```

<h4><li><a href="https://www.geeksforgeeks.org/calculating-n-th-real-root-using-binary-search/">
N-th root of a number
</a></li></h4>

```
given: x, n
approach: binary search

low = 0
high = 0

if x >= 0 AND x <= 1
low = x
high = 1

else
low = 1
high = x

<- used for taking approximations of the answer ->
epsilon = 0.00000001

<- Do binary search ->
guess = (low + high) / 2

WHILE abs((pow(guess, n)) - x) >= epsilon

    if pow(guess, n) > x
        high = guess
    else
        low = guess

    guess = (low + high) / 2

n-th root of a number: guess

```

<h4><li><a href="https://www.geeksforgeeks.org/find-median-row-wise-sorted-matrix/">
Find median in row wise sorted matrix
</a></li></h4>

```

approach: binary search
max = MIN_VALUE
min = MAX_VALUE

i:0 -> r - 1
<- Finding the minimum element ->
if m[i][0] < min
min = m[i][0]

    <- Finding the maximum element ->
    if(m[i][c-1] > max)
        max = m[i][c-1]

desired = (r \* c + 1) / 2

WHILE min < max

    mid = min + (max - min) / 2
    place = 0
    get = 0

    <- Find count of elements smaller than mid ->
    i: 0 -> r - 1
        get = binarySearch(m[i],mid)

        <- If element not found, RETURNs -(insertion_point) - 1 ->
        if(get < 0)
            get = abs(get) - 1

        <- If element is found, RETURNs index ->
        else
            WHILE get < m[i].length AND m[i][get] == mid
                get += 1

        place += get


    if place < desired
        min = mid + 1

    else
        max = mid

median: min

```

<h4><li><a href="https://leetcode.com/problems/single-element-in-a-sorted-array/">
Single Element in a Sorted Array
</a></li></h4>

```

approach: binary search

lo = 0
hi = nums.length - 1

WHILE lo < hi
mid = (lo + hi) / 2
temp = mid ^ 1

    if nums[mid] == nums[temp]
        lo = mid + 1
    else
        hi = mid

search index: lo

```

<h4><li><a href="https://leetcode.com/problems/search-in-rotated-sorted-array/">
Search in Rotated Sorted Array
</a></li></h4>

```
approach: binary search

lo = 0
hi = n - 1

WHILE lo < hi

    mid = (lo + hi) / 2

    <- target and mid are on the same side ->
    if nums[mid]-nums[n-1] * target-nums[n-1] > 0

        if(nums[mid] < target)
            lo = mid + 1
        else
            hi = mid

    <- target on the left side ->
    ELIF target > nums[n-1])
        hi = mid

    <- target on the right side ->
    else
        lo = mid + 1

search index: nums[lo] == target ? lo : -1
```

<h4><li><a href="https://leetcode.com/problems/median-of-two-sorted-arrays/">
Median of Two Sorted Arrays
</a></li></h4>

```
findMedianSortedArrays (A, B)
    m = A.length, n = B.length
    l = (m + n + 1) / 2
    r = (m + n + 2) / 2
    RETURN (getkth(A, 0, B, 0, l) + getkth(A, 0, B, 0, r)) / 2.0

getkth(A, aStart, B, bStart, k)
    if aStart > A.length - 1
        RETURN B[bStart + k - 1]

    if bStart > B.length - 1
        RETURN A[aStart + k - 1]

    if k == 1
        RETURN min(A[aStart], B[bStart])

    aMid = Integer.MAX_VALUE
    bMid = Integer.MAX_VALUE

    if aStart + k/2 - 1 < A.length
        aMid = A[aStart + k/2 - 1]

    if bStart + k/2 - 1 < B.length
        bMid = B[bStart + k/2 - 1]

    <- Check: aRight + bLeft ->
    if aMid < bMid
        RETURN getkth(A, aStart + k/2, B, bStart, k - k/2)

    <- Check: bRight + aLeft ->
    else
        RETURN getkth(A, aStart, B, bStart + k/2, k - k/2)
```

<h4><li><a href="https://www.geeksforgeeks.org/allocate-minimum-number-pages/">
FindAllocate minimum number of pages
</a></li></h4>

```
GIVEN & CALL:
arr1[5] = {2, 3, 6, 7, 9}
arr2[4] = {1, 4, 8, 10}
kth(arr1, arr2, arr1 + 5, arr2 + 4, k - 1)

kth (arr1, arr2, end1, end2, k)
    IF arr1 == end1
        RETURN arr2[k]

    IF arr2 == end2
    	RETURN arr1[k]

    mid1 = (end1 - arr1) / 2
    mid2 = (end2 - arr2) / 2

    IF mid1 + mid2 < k
    	IF arr1[mid1] > arr2[mid2]
    		RETURN kth(arr1, arr2 + mid2 + 1, end1, end2, k - mid2 - 1)
        ELSE
    		RETURN kth(arr1 + mid1 + 1, arr2, end1, end2, k - mid1 - 1)

    ELSE
    	IF arr1[mid1] > arr2[mid2]
    		RETURN kth(arr1, arr2, arr1 + mid1, end2, k)
    	ELSE
    		RETURN kth(arr1, arr2, end1, arr2 + mid2, k)
```

<h4><li><a href="https://www.spoj.com/problems/AGGRCOW/">
Aggressive cows
</a></li></h4>

```

```

<h4><li><a href="https://leetcode.com/problems/search-a-2d-matrix/">
Power Set
</a></li></h4>

```

approach: bit manipulation

List<List<Integer>> powerSet

i: 0 -> (1 << nums.length)
List<Integer> set

    j: 0 -> nums.length - 1
        if (i >> j) & 1 != 0
            set.add(nums[j])

    powerSet.add(set)

power set: powerSet

```

<h4><li><a href="https://www.geeksforgeeks.org/stack-data-structure-introduction-program/">
Implementing Stack using Arrays
</a></li></h4>

```

Stack
{
<- Maximum size of Stack, MAX = 1000 ->
top
a[MAX]

    Stack
        top = -1


    isEmpty
        RETURN (top < 0)


    push(x)
        if top >= (MAX - 1)
            sout("Stack Overflow")
            RETURN false

        else
            a[++top] = x
            sout(x + " pushed into stack")
            RETURN true


    pop
        if top < 0
            sout("Stack Underflow")
            RETURN 0

        else
            RETURN a[top--]


    peek
        if top < 0
            sout("Stack Underflow")
            RETURN 0

        else
            RETURN a[top]

    print
        i = top -> 0
            sout(a[i])

}

```

<h4><li><a href="https://www.geeksforgeeks.org/array-implementation-of-queue-simple/">
Implement Queue using Arrays
</a></li></h4>

```
Queue
{
    front, rear, capacity
    int queue[]

    Queue (c)
    	front = rear = 0
    	capacity = c
    	queue = new int[capacity]


    enqueue (data)
    	if capacity == rear
    		sout("Queue is full")
    		RETURN

    	else
    		queue[rear] = data
    		rear++


    dequeue
    	if front == rear
    		sout("Queue is empty")

    	<- shift all the elements from index 2 till rear to the right by one ->
    	else
    		i: 0 -> rear - 2
    			queue[i] = queue[i + 1]

    		<- store 0 at rear indicating there's no element ->
    		if (rear < capacity)
    			queue[rear] = 0

    		rear--


    display
    	if front == rear
    		sout("\nQueue is Empty\n")
    		RETURN

    	i: front -> rear - 1
    		sout(queue[i])


    front
    	if front == rear
    		sout("Queue is Empty")
    	else
            sout(queue[front])

}

```

<h4><li><a href="https://leetcode.com/problems/implement-stack-using-queues/">
Implement Stack using Queues
</a></li></h4>

```
MyStack {

    LinkedList<Integer> q1


    push (x)
        q1.add(x)
        sz = q1.size
        WHILE sz > 1
            q1.add(q1.remove)
            sz--


    pop
        q1.remove


    top
        RETURN q1.peek


    empty
        RETURN q1.isEmpty

}

```

<h4><li><a href="https://leetcode.com/problems/implement-queue-using-stacks/">
Implement Queue using Stacks
</a></li></h4>

```

approach name: 0(1) amortised method

MyQueue
{
Stack<Integer> s1
Stack<Integer> s2

    <- Push element x to the back of queue ->
    push (x)
        if s1.isEmpty
            front = x

        s1.push(x)

    <- Removes the element from in front of queue ->
    pop
        if s2.isEmpty
            WHILE !s1.isEmpty
                s2.push(s1.pop)

        s2.pop


    peek
        if !s2.isEmpty
            RETURN s2.peek

        RETURN front


    empty
        RETURN s1.isEmpty AND s2.isEmpty

}

```

<h4><li><a href="https://leetcode.com/problems/valid-parentheses/">
Valid Parentheses
</a></li></h4>

```

approach: stacks

Stack<Character> stack

FOR (c : s.toCharArray)
    IF c == '('
    stack.push(')')

    ELIF c == '{'
        stack.push('}')

    ELIF c == '['
        stack.push(']')

    ELIF stack.isEmpty OR stack.pop != c
        RETURN false

RETURN stack.isEmpty

```

<h4><li><a href="https://leetcode.com/problems/next-greater-element-i/">
Next Greater Element I
</a></li></h4>

```
approach: stacks

<- Map from x to next greater element of x ->
Map<Integer, Integer> map
Stack<Integer> stack

FOR (num : nums)
    WHILE !stack.isEmpty AND stack.peek < num
        map.put(stack.pop, num)
    stack.push(num)

i: 0 -> findNums.length - 1
    findNums[i] = map.getOrDefault(findNums[i], -1)

next greater element array: findNums
```

<h4><li><a href="https://www.geeksforgeeks.org/next-smaller-element/">
Next Smaller Element
</a></li></h4>

```
approach: stacks

Stack<Integer> s
HashMap<Integer, Integer> mp

    FOR (num : nums)
        WHILE !s.isEmpty AND s.peek > num
            mp.put(s.pop, num)

    s.push(num)

i: 0 -> findNums.length - 1
    findNums[i] = map.getOrDefault(findNums[i], -1)

next smaller element array: findNums
```

<h4><li><a href="https://leetcode.com/problems/find-the-duplicate-number/">
Find the Duplicate Number
</a></li></h4>

[**58. Sort a stack using recursion**](https://www.geeksforgeeks.org/sort-a-stack-using-recursion/)

```

sortedInsert (s, x)
{
    if s.isEmpty OR x > s.peek
        s.push(x)
        RETURN

    <- If top is greater, remove the top item and recur ->
    temp = s.pop
    sortedInsert(s, x)

    <- Put back the top item removed earlier ->
    s.push(temp)
}

<- Method to sort stack ->
sortStack(s)
{
    if !s.isEmpty
    <- Remove the top item ->
        int x = s.pop

        <- Sort remaining stack ->
        sortStack(s)

        <- Push the top item back in sorted stack ->
        sortedInsert(s, x)
}
```

<h4><li><a href="https://www.geeksforgeeks.org/lru-cache-implementation/">
LRU Cache Implementation
</a></li></h4>

```

LRUCache
{
    Deque<Integer> doublyQueue
    HashSet<Integer> hashSet
    int CACHE_SIZE

    LRUCache (capacity)
    	CACHE_SIZE = capacity


    refer (page)
    	if (!hashSet.contains(page))
    		if doublyQueue.size == CACHE_SIZE
    			last = doublyQueue.removeLast
    			hashSet.remove(last)
        else
    		doublyQueue.remove(page)

    	doublyQueue.push(page)
    	hashSet.add(page)


    display
    	Iterator<Integer> itr
    	WHILE itr.hasNext
    		sout(itr.next)
}
```

<h4><li><a href="https://leetcode.com/problems/largest-rectangle-in-histogram/">
Largest Rectangle in Histogram
</a></li></h4>

```

<- idx of the first bar the left that is lower than current ->
int[n] lessFromLeft

<- idx of the first bar the right that is lower than current ->
int [n] lessFromRight

lessFromRight[n - 1] = n
lessFromLeft[0] = -1

i: 1 -> n - 1
    p = i - 1
    WHILE p >= 0 AND height[p] >= height[i]
        p = lessFromLeft[p]

    lessFromLeft[i] = p

i: n - 2 -> 0
    p = i + 1
    WHILE p < n AND height[p] >= height[i]
        p = lessFromRight[p]

    lessFromRight[i] = p

maxArea = 0

i: 0 -> n - 1
    maxArea = max(maxArea, height[i] \* (lessFromRight[i] - lessFromLeft[i] - 1))

RETURN maxArea
```

<h4><li><a href="https://leetcode.com/problems/sliding-window-maximum/">
Sliding Window Maximum
</a></li></h4>

```
given: int[] a, int k

if a == null OR k <= 0
    return new int[0]

n = a.length
int[n - k + 1] r
ri = 0

<- store index ->
Deque<Integer> q

i: 0 -> n - 1
    <- remove numbers out of range k ->
    WHILE !q.isEmpty() AND q.peek() < i - k + 1
        q.poll

            <- remove smaller numbers in k range as they are useless ->
            WHILE !q.isEmpty AND a[q.peekLast()] < a[i]
                q.pollLast

            <- q contains index... r contains content ->
            q.offer(i)
            if i >= k - 1
                r[ri++] = a[q.peek]


RETURN r
```

<h4><li><a href="https://leetcode.com/problems/min-stack/">
Min Stack
</a></li></h4>

```
MinStack
{
    Node head

    Node
    {
        int val
        int min
        Node next

        Node (val, min, next)
            this.val = val
            this.min = min
            this.next = next
    }


    push (x)
    {
        if head == null
            head = new Node(x, x, null)
        else
            head = new Node(x, min(x, head.min), head)
    }


    pop
    {
        head = head.next
    }


    top
    {
        RETURN head.val
    }


    getMin
    {
        RETURN head.min
    }

}
```

<h4><li><a href="https://leetcode.com/problems/rotting-oranges/">
Rotting Oranges
</a></li></h4>

```

given:
    0: an empty cell
    1: a fresh orange
    2: a rotten orange

rows = grid.length
cols = grid[0].length
Queue<int[]> queue
count_fresh = 0

<- Put the position of all rotten oranges in queue count the number of fresh oranges ->
i: 0 -> rows - 1
    j: 0 -> cols - 1
        if grid[i][j] == 2
            queue.offer({i , j})

        else if(grid[i][j] == 1) {
            count_fresh++

//if count of fresh oranges is zero --> RETURN 0

if count_fresh == 0
    RETURN 0

count = 0

dirs = [[1,0], [-1,0], [0,1], [0,-1]]

<- bfs starting from initially rotten oranges ->
WHILE !queue.isEmpty
    count++
    size = queue.size

    i: 0 -> size - 1
        point = queue.poll

        for (dir[] : dirs)
            x = point[0] + dir[0]
            y = point[1] + dir[1]

            if(x < 0 || y < 0 || x >= rows || y >= cols || grid[x][y] == 0 || grid[x][y] == 2)
                continue

            <- mark the orange as rotten ->
            grid[x][y] = 2
            queue.offer([x , y])
            count_fresh--

RETURN count_fresh == 0 ? count - 1 : -1
```

<h4><li><a href="https://www.geeksforgeeks.org/z-algorithm-linear-time-pattern-searching-algorithm/">
Z algorithm (Linear time pattern searching Algorithm)
</a></li></h4>

```

search (text, pattern)
concat = pattern + "$" + text
l = concat.length

    int[l] Z

    getZarr(concat, Z)

    i: 0 -> l - 1
        if(Z[i] == pattern.length
            sout(i - pattern.length - 1)

getZarr (str, Z)
n = str.length
L = 0
R = 0

    i: 1 -> n - 1
        if i > R
            L = R = i

            WHILE R < n && str[R - L] == str[R]
                R++

            Z[i] = R - L
            R--

        else
            k = i - L

            if Z[k] < R - i + 1
                Z[i] = Z[k]
            else
                L = i
                WHILE R < n && str[R - L] == str[R]
                    R++

                Z[i] = R - L
                R--

```

<h4><li><a href="https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/">
KMP Algorithm for Pattern Searching
</a></li></h4>

```

KMPSearch (pat, txt)
{
    int M = pat.length
    int N = txt.length

    int[M] lps
    j = 0

    computeLPSArray(pat, M, lps)

    i = 0
    WHILE (i < N)
        if pat[j] == txt[i]
            j++
            i++

        if j == M
            sout((i - j)
            j = lps[j - 1]

        ELIF i < N AND pat[j] != txt[i]
            if (j != 0)
                j = lps[j - 1]
            else
                i = i + 1
}

computeLPSArray(pat, M, lps)
{
    len = 0
    i = 1
    lps[0] = 0

    WHILE i < M
        if pat[i] == pat[len]
            len++
            lps[i] = len
            i++

        else
            if len != 0
                len = lps[len - 1]

            else
                lps[i] = len
                i++
}
```

<h4><li><a href="https://www.geeksforgeeks.org/minimum-characters-added-front-make-string-palindrome/">
Minimum characters to be added at front to make string palindrome
</a></li></h4>

```

approach: lps array of KMP algorithm

computeLPSArray (str)
n = str.length
int[n] lps
i = 1
len = 0

    lps[0] = 0

    WHILE i < n
    	if str[i] == str[len]
    		len++
    		lps[i] = len
    		i++

        else
    		if len != 0
    			len = lps[len - 1]

    		else
    			lps[i] = 0
    			i++


    RETURN lps

getMinCharToAddedToMakeStringPalin (str)
StringBuilder s
s.append(str)

    String rev = s.reverse.toString
    s.reverse.append("$").append(rev)

    lps = computeLPSArray(s.toString)

    RETURN str.length - lps[s.length - 1]

```

<h4><li><a href="https://www.geeksforgeeks.org/check-whether-two-strings-are-anagram-of-each-other/">
Check whether two strings are anagram of each other
</a></li></h4>

```
int[256] count
int i

if str1.length != str2.length
    RETURN false

i: 0 -> str1.length - 1
    count[str1[i] - 'a']++
    count[str2[i] - 'a']--

i: 0 -> 256 - 1
    if count[i] != 0
        RETURN false

RETURN true
```

<h4><li><a href="https://leetcode.com/problems/count-and-say/">
Find the Duplicate Number
</a></li></h4>

```

countAndSay (n)
s = "1"

    i: 1 -> n - 1
        s = countIdx(s)

    RETURN s

countIdx (s)
StringBuilder sb
c = s[0]
count = 1

    i: 1 -> s.length - 1
        if s[i] == c
            count++

        else
            sb.append(count)
            sb.append(c)
            c = s[i]
            count = 1

    sb.append(count)
    sb.append(c)

    RETURN sb.toString

```

<h4><li><a href="https://leetcode.com/problems/compare-version-numbers/">
Compare Version Numbers
</a></li></h4>

```
given: String version1, String version2

levels1 = version1.split("\\.")
levels2 = version2.split("\\.")

length = Math.max(levels1.length, levels2.length)

i: 0 -> length - 1
    v1 = i < levels1.length ? Integer.parseInt(levels1[i]) : 0
    v2 = i < levels2.length ? Integer.parseInt(levels2[i]) : 0

    compare = v1.compareTo(v2)

    IF compare != 0
        return compare


return 0
```

<h4><li><a href="https://leetcode.com/problems/compare-version-numbers/">
Compare Version Numbers
</a></li></h4>

```

```

</ol>
