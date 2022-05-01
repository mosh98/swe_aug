import spacy_udpipe
import random
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath
import nltk
#!pip install spacy-udpipe
#!pip install nltk

#TODO: Fix calling function
class type_DA():
    def __init__(self, word_vec_path):
        spacy_udpipe.download("sv")
        nltk.download('stopwords')
        self.nlp = spacy_udpipe.load("sv")
        self.stop_words_ = set(nltk.corpus.stopwords.words('swedish'))
        self.wv_from_text = KeyedVectors.load_word2vec_format(datapath(word_vec_path), binary=False) #link need to be fixed


    def get_type(self, text):
        """
        Returns the POS type of the text.

        @:param text: The text to be analyzed.
        @:return: list of pos_tags
        """

        doc = self.nlp(text)

        pos_list = []

        for token in doc:
            pos_list.append(token.pos_)


        return pos_list

    def synonym_replacement_vec_type(self,words, n_sr, typ):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words_]))
        random.shuffle(random_word_list)

        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms_vec(random_word)  # self
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                type_is = self.get_type(synonym)

                if type_is[0] == typ:

                    new_words = [synonym if word.lower()  == random_word else word for word in new_words]
                    num_replaced += 1
                else:
                    continue
            if num_replaced >= n_sr:
                break

            # this is stupid but we need it, trust me

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words

    def get_synonyms_vec(self,word):
        synonyms = set()
        flag = False
        vec = None
        try:
            vec = self.wv_from_text.similar_by_word(word.lower())
        except KeyError:
            flag = True
            pass

        if flag is False:
            synonyms.add(vec[0][0])

        if word in synonyms:
            synonyms.remove(word)

        return synonyms

    def type_synonym_sr(self, text, token_type = None, n = 1):
        """
        Returns the POS type of the text.
        :param text:
        :param token_type: if no type is given, a random type is chosen
        :param n: number of sentences you want as output
        :return:
        """

        if token_type not in ["NOUN", "VERB", "ADJ", "ADV", "PROPN","CONJ"]:
            token_type = None
        sentences = []
        words = text.split(' ')
        for i in range(n):
            if token_type is None:
                token_type = random.choice(['ADJ', 'VERB', 'NOUN', 'ADV'])
                sen = self.synonym_replacement_vec_type(words, n, token_type)
                sentences.append(sen)
            else:
                sen = self.synonym_replacement_vec_type(words, n, token_type)
                sentences.append(sen)


        return sentences