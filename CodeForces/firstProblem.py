def getInput():
    n = int(input())
    words = []
    for i in range(n):
        words.append(input())
    return words


if __name__ == '__main__':
    words = getInput()
    result = []
    for word in words:
        n = len(word)
        if n > 10:
            print(word[0] + str(n-2) + word[-1])
        else:
            print(word)