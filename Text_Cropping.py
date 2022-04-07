import math

class cropper():
    def __init__(self):
        self.percentage = 0.25
        pass
    #https://colab.research.google.com/drive/1oQkPlCREnhiAz0zTifgVi0YWcE6hIzi6#scrollTo=EFMPJVBTNJNh

    def text_fragmeter(self, sentence):

        framented = []
        words = sentence.split(" ")
        sen_length = len(words)
        chunk_length = sen_length * self.percentage

        for i in range(0, sen_length, chunk_length):
            #print(words[i:i + n])
            framented.append(words[i:i + chunk_length])

        return framented



