'''
Reverse linked list :: recursively

'''

class Solution:
    def reverseList(self, node: ListNode) -> ListNode:
        if not node or not node.next:
            return node

        p = self.reverseList(node.next)
        node.next.next = node
        node.next = None

        return p

'''
Reverse linked list :: iteratively
Better because space O(1)

'''
class Solution:
    def reverseList(self, node: ListNode) -> ListNode:
        prev = None

        while node:

            nextT = node.next
            node.next = prev
            prev = node
            node = nextT


        return prev

'''
Reverse linked list into k groups

https://leetcode.com/problems/reverse-nodes-in-k-group/submissions/
'''

class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        def reverse(node, count):

            prev = None
            while count > 0:
                nextT = node.next
                node.next = prev
                prev = node
                node = nextT
                count -= 1

            return node, prev

        node, count = head, 0
        while node and count < k:
            node = node.next
            count += 1

        if count < k:
            return head

        new_head, prev = reverse(head, count)
        # head gets reversed with prev, so head.next needs to be pointing to next reversed group's prev
        head.next = self.reverseKGroup(new_head, k)

        return prev

'''
Find middle of linked list

Time: O(N)
Space: O(1)

'''
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        if not head:
            return None

        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow

'''
Rotate a linked list

Time: O(N)
Space: O(1)

'''


class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return None

        def getLength(node):
            count = 1

            while node.next != None:
                node = node.next
                count += 1

            return node, count

        tail, length = getLength(head)
        k = k % length
        if k == 0:
            return head

        start, newHeadPostion, count = head, length - k % length - 1, 0

        while count < newHeadPostion:
            start = start.next
            count += 1

        tail.next = head
        head = start.next
        start.next = None

        return head

'''
Find intersection of linked list
https://leetcode.com/problems/linked-list-cycle/

Time = O(M+N)
Space = O(1)
'''


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None

        def getLength(node):
            count = 1
            while node:
                node = node.next
                count += 1
            return count

        lenA = getLength(headA)
        lenB = getLength(headB)

        while lenA > lenB:
            headA = headA.next
            lenA -= 1
        while lenB > lenA:
            headB = headB.next
            lenB -= 1

        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next

        return None

'''
Find intersection Node of linked list

https://leetcode.com/problems/linked-list-cycle-ii/
'''


class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head is None:
            return None

        def hasCycle(head):
            slow, fast = head, head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                if slow == fast:
                    return slow

            return None

        intersect = hasCycle(head)
        if intersect is None:
            return None

        ptr1 = head
        ptr2 = intersect

        while ptr1 != ptr2:
            ptr2 = ptr2.next
            ptr1 = ptr1.next

        return ptr1

'''
Nth node from back 

https://leetcode.com/problems/remove-nth-node-from-end-of-list/

'''

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:
            return None

        dummy = ListNode(0, head)
        first, second = dummy, dummy
        for i in range(n + 1):
            first = first.next

        while first:
            first = first.next
            second = second.next

        second.next = second.next.next

        return dummy.next

'''
Swap Nodes in pairs

https://leetcode.com/problems/swap-nodes-in-pairs/

'''


class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head:
            return None

        p = ListNode(0, head)
        answer = p
        q, r = head, head.next

        while r:
            p.next = r
            q.next = r.next
            r.next = q

            p = q
            q = p.next
            r = q.next if q else None

        return answer.next

'''
Add two numbers

https://leetcode.com/problems/add-two-numbers-ii/

'''


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1

        def fillStack(node):
            stack = []
            while node:
                stack.append(node.val)
                node = node.next

            return stack

        s1 = fillStack(l1)
        s2 = fillStack(l2)

        Node = None
        carry = 0
        while s1 or s2:
            if s1 and s2:
                newVal = s1.pop() + s2.pop() + carry
            elif s1 and not s2:
                newVal = s1.pop() + carry
            elif s2 and not s1:
                newVal = s2.pop() + carry

            carry = newVal // 10
            newVal %= 10

            thisNode = ListNode(newVal, Node)
            Node = thisNode

        if carry > 0:
            thisNode = ListNode(carry, Node)
            Node = thisNode

        return Node

'''
Delete node without the head being given

https://leetcode.com/problems/delete-node-in-a-linked-list/

'''
class Solution:
    def deleteNode(self, node):
        if not node:
            return None

        prev = None
        while node.next:
            node.val = node.next.val
            prev = node
            node = node.next

        prev.next = None

'''
https://leetcode.com/problems/swap-nodes-in-pairs/

'''

'''
Palindrome Linked List
O(N) time and O(N) space

'''


class Solution:
    def isPalindrome(self, head: ListNode) -> bool:

        def reverse(node):
            prev = None
            while node:
                tempNext = node.next
                node.next = prev
                prev = node
                node = tempNext
            return prev

        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next

        head2 = reverse(slow)
        print(head.val, head2.val)
        while head and head2:
            if head.val != head2.val:
                return False
            head = head.next
            head2 = head2.next

        return True


'''
Plus one to linked list
https://leetcode.com/problems/plus-one-linked-list/solution/

'''


class Solution:
    def plusOne(self, head: ListNode) -> ListNode:

        pointer = head
        h = nonZero = ListNode(0)
        nonZero.next = pointer

        while pointer:
            if pointer.val != 9:
                nonZero = pointer
            pointer = pointer.next

        nonZero.val += 1
        nonZero = nonZero.next

        while nonZero:
            nonZero.val = 0
            nonZero = nonZero.next

        return h if h.val else h.next