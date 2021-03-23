def getInput():
    n = int(input())
    views = []
    for i in range(n):
        views.append(list(map(int, input().split())))
    return views

if __name__ == '__main__':
    views = getInput()
    probs = 0
    for view in views:
        if sum(view) > 1:
            probs += 1
    print(probs)
