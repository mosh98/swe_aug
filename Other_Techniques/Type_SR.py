import spacy_udpipe

#!pip install spacy-udpipe
#!pip install nltk

class type_DA():
    def __init__(self):
        spacy_udpipe.download("sv")
        self.nlp = spacy_udpipe.load("sv")

    def get_type(self, text):
        """
        Returns the POS type of the text.

        @:param text: The text to be analyzed.
        @:return: list of pos_tags
        """

        doc = self.nlp(text)

        pos_list = []
        tag_list = []
        for token in doc:
            pos_list.append(token.pos_)
            tag_list.append(token.tag_)

        return pos_list, tag_list
