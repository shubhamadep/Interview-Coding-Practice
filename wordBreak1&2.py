'''
Word Break 1

'''


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        memo = collections.defaultdict(list)
        wordDict = set(wordDict)

        def helper(word):

            if not word:
                return True

            if word in memo:
                return memo[word]

            for end in range(1, len(word) + 1):
                current_word = word[:end]
                if current_word in wordDict:
                    val = helper(word[end:])
                    memo[word] = val
                    if val:
                        return True

            return False

        return helper(s)

'''
Word Break 2

'''


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:

        memo = collections.defaultdict(list)
        wordDict = set(wordDict)

        def find_sentences(sentence):

            if not sentence:
                return [[]]

            if sentence in memo:
                return memo[sentence]

            for sentence_end in range(1, len(sentence) + 1):
                current_word = sentence[:sentence_end]
                if current_word in wordDict:
                    sentence_breaks = find_sentences(sentence[sentence_end:])

                    for sentence_break in sentence_breaks:
                        memo[sentence].append([current_word] + sentence_break)

            return memo[sentence]

        find_sentences(s)
        return [" ".join(sentence) for sentence in memo[s]]
