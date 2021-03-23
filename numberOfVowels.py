class Solution:
    def numberofVowels(self, wordLength, consVowels):
        self.count = 0
        return self.helper(wordLength, consVowels, 0, 0)

    def helper(self, wordLength, consVowels, index, remainingVowels):

        if index == wordLength:
            self.count += 1
            return

        self.helper(wordLength, consVowels, index+1, remainingVowels-1)
        self.helper(wordLength, consVowels, index+1, remainingVowels)