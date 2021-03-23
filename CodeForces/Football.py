import sys

def getInput():
    return list(map(str, input().split()))

if __name__ == '__main__':
    inputs = getInput()
    if not inputs:
        print("NO")
        sys.exit(0)

    count = 0
    curr = 0
    for input in list(inputs[0]):
        if curr == input:
            count += 1
        else:
            curr = input
            count = 1
        if count == 7:
            print("YES")
            sys.exit(0)

    print("NO")