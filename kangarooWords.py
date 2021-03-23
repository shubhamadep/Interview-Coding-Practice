class Solution:
    def isKangarooWord(self, words, synonyms, antonyms)):
        if not synonyms and not antonyms:
            return 0

        score = 0
        seen= set()
        seen = collections.defaultdict()
        for synonym in synonyms:
            word, syns = synonym.split(":")
            if word not in words:
                continue
            for syn in syns.split(","):
                if subsequence(syn, word):
                    score += 1
                    if syn not in seen:
                        seen.add(syn)
                    else:
                        score += 1
                        seen.remove(syn)

        for antonym in antonyms:
            word, antons = antonym.split(":")
            if word not in words:
                continue
            for ant in antons.split(","):
                if subsequence(ant, word):
                    score -= 1

        return score

    def subsequence(self, s, t):

        LEFT_BOUND, RIGHT_BOUND = len(s), len(t)

        p_left = p_right = 0

        while p_left < LEFT_BOUND and p_right < RIGHT_BOUND:
            if s[p_left] == t[p_right]:
                p_left += 1

            p_right += 1

        return p_left == LEFT_BOUND and s not in t