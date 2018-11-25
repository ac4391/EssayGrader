import re

class Essay:

    def __init__(self, text, lowercase=True):
        # Convert essay string into a list of strings where each string is a word
        if lowercase == True:
            text_lower = text.lower()
            self.words = re.findall(r"[\w']+|[.,!?;]", text_lower)

        else:
            self.words = re.findall(r"[\w']+|[.,!?;]", text)

        self.word_count = len(self.words)

        # Count word occurrences in the essay
        self.word_dict={}
        for word in self.words:
            if word in self.word_dict:
                self.word_dict[word] += 1
            else:
                self.word_dict[word] = 1

        # The list of unique words is a list of the keys in the word dictionary
        self.unique_words = [key for key in self.word_dict.keys()]
