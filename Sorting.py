'''
bubble sort

Time: O(n^2)
Space: O(1)

'''
def bubbleSort(array):
    isSorted = False
    counter = 0

    while not isSorted:
        isSorted = True
        for i in range(1, len(array) - counter):
            if array[i] < array[i - 1]:
                array[i], array[i - 1] = array[i - 1], array[i]
                isSorted = False
        counter += 1

    return array

'''
insertion sort

Time: O(n^2)
Space: O(1)

'''

def insertionSort(array):
    for i in range(1, len(array)):
        for j in reversed(range(1, i + 1)):
            if array[j] < array[j - 1]:
                array[j], array[j - 1] = array[j - 1], array[j]

    return array

'''
Quick sort

Time:

    Worst: O(n^2)
    Best: O(nlogn)
    Avg: O(nlogn)

Space: O(logn) 

'''
def quickSort(array):
    helper(array, 0, len(array) - 1)
	return array

def helper(array, startIdx, endIdx):
	if startIdx >= endIdx:
		return
	pivot = startIdx
	left = startIdx + 1
	right = endIdx
	while right >= left:
		if array[left] > array[pivot] and array[right] < array[pivot]:
			swap(array, left, right)
		if array[left] <= array[pivot]:
			left += 1
		if array[right] >= array[pivot]:
			right -= 1
	swap(array, pivot, right)
	leftLength = right - 1 - startIdx > endIdx - ( right + 1)
	if leftLength:
		helper(array, startIdx, right - 1)
		helper(array, right+1, endIdx)
	else:
		helper(array, right+1, endIdx)
		helper(array, startIdx, right - 1)

def swap(array, left, right):
	array[left], array[right] = array[right], array[left]

'''
Heap Sort

Time: O(logn)
Space: O(1)

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
Merge sort

Time: O(logn)
Space: O(nlogn)

for space: O(n) algorithm, watch the video≠≠≠≠

'''


def mergeSort(array):
    if len(array) == 1:
        return array

    mid = len(array) // 2
    return mergeSortArrays(mergeSort(array[:mid]), mergeSort(array[mid:]))


def mergeSortArrays(left, right):
    sortedArray = [None] * (len(left) + len(right))
    k = i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            sortedArray[k] = left[i]
            i += 1
        else:
            sortedArray[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        sortedArray[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        sortedArray[k] = right[j]
        j += 1
        k += 1

    return sortedArray


