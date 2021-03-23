'''
continuous mean



'''
class ContinuousMedianHandler:
    def __init__(self):
        # Write your code here.
        self.median = None
        self.lowerHalf = Heap(MAX_HEAP_FUNC, [])
        self.greaterHalf = Heap(MIN_HEAP_FUNC, [])


def insert(self, number):
    if not self.lowerHalf.length or self.lowerHalf.peek() > number:
        self.lowerHalf.insert(number)
    else:
        self.greaterHalf.insert(number)
    self.rebalance()
    self.updateMedian()
    print(self.median)


def updateMedian(self):
    if self.lowerHalf.length == self.greaterHalf.length:
        self.median = (self.lowerHalf.peek() + self.greaterHalf.peek()) / 2
    elif self.lowerHalf.length > self.greaterHalf.length:
        self.median = self.lowerHalf.peek()
    else:
        self.median = self.greaterHalf.peek()


def rebalance(self):
    if self.lowerHalf.length - self.greaterHalf.length == 2:
        self.greaterHalf.insert(self.lowerHalf.remove())
    elif self.greaterHalf.length - self.lowerHalf.length == 2:
        self.lowerHalf.insert(self.greaterHalf.remove())


def getMedian(self):
    return self.median


class Heap:
    def __init__(self, comparatorFunction, array):
        self.comparisonFunction = comparatorFunction
        self.heap = self.buildHeap(array)
        self.length = len(self.heap)

    def buildHeap(self, array):
        firstParentIdx = (len(array) - 2) // 2
        for currentIdx in reversed(range(firstParentIdx + 1)):
            self.siftDown(currentIdx, len(array) - 1, array)
        return array

    def peek(self):
        return self.heap[0]

    def insert(self, value):
        self.heap.append(value)
        self.length += 1
        self.siftUp(self.length - 1, self.heap)

    def remove(self):
        self.swap(0, len(self.heap) - 1, self.heap)
        value = self.heap.pop()
        self.length -= 1
        self.siftDown(0, self.length - 1, self.heap)
        return value

    def siftUp(self, index, heap):
        parent = (index - 1) // 2
        while index > 0:
            if self.comparisonFunction(heap[index], heap[parent]):
                self.swap(parent, index, heap)
                index = parent
                parent = (index - 1) // 2
            else:
                return

    def siftDown(self, startIdx, endIdx, heap):

        childOneIdx = startIdx * 2 + 1
        while childOneIdx <= endIdx:
            childTwoIdx = startIdx * 2 + 2 if startIdx * 2 + 2 <= endIdx else -1
            if childTwoIdx != -1:
                if self.comparisonFunction(heap[childTwoIdx], heap[childOneIdx]):
                    toSwap = childTwoIdx
                else:
                    toSwap = childOneIdx
            else:
                toSwap = childOneIdx
            if self.comparisonFunction(heap[toSwap], heap[startIdx]):
                self.swap(toSwap, startIdx, heap)
                startIdx = toSwap
                childOneIdx = startIdx * 2 + 1
            else:
                return

    def swap(self, value1, value2, heap):
        heap[value1], heap[value2] = heap[value2], heap[value1]


def MAX_HEAP_FUNC(a, b):
    return a > b


def MIN_HEAP_FUNC(a, b):
    return a < b


'''
HeapSort

'''


def heapSort(array):
    heap = buildHeap(array)
    for endIdx in reversed(range(1, len(array))):
        swap(array, endIdx, 0)
        siftDown(array, 0, endIdx - 1)

    return array


def buildHeap(array):
    parent = (len(array) - 2) // 2
    for currentIdx in reversed(range(parent + 1)):
        siftDown(array, currentIdx, len(array) - 1)


def siftDown(array, start, end):
    childOneIdx = 2 * start + 1
    while childOneIdx <= end:
        childTwoIdx = 2 * start + 2 if 2 * start + 2 <= end else -1
        if childTwoIdx != -1 and array[childTwoIdx] > array[childOneIdx]:
            toSwap = childTwoIdx
        else:
            toSwap = childOneIdx
        if array[start] < array[toSwap]:
            swap(array, start, toSwap)
            start = toSwap
            childOneIdx = 2 * start + 1
        else:
            return


def swap(array, value1, value2):
    array[value1], array[value2] = array[value2], array[value1]

'''
V Important

Reorganizing string such that no two characters are adjacent

https://leetcode.com/problems/reorganize-string/

'''
import heapq

class Solution:
    def reorganizeString(self, S: str) -> str:

        counter = collections.Counter(S)
        h = [(-counter[x], x) for x in set(S)]

        heapq.heapify(h)
        result = []

        while len(h) >= 2:

            c1, char1 = heapq.heappop(h)
            c2, char2 = heapq.heappop(h)
            result.extend([char1, char2])
            if c1 + 1: heapq.heappush(h, (c1 + 1, char1))
            if c2 + 1: heapq.heappush(h, (c2 + 1, char2))

        print(h, result)

        #only one element is remaining and its count is -1 because its a max heap
        if len(h) > 0:
            return "".join(result) + h[0][1] if h[0][0] == -1 else ""
        else:
            return "".join(result)

'''
kth largest element in a stream

'''
import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):

        self.heap = []
        self.k = k
        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, val)

        return self.heap[0]
