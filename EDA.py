import pandas as pd
import numpy as np
import nltk
import random
import spacy_udpipe
import re
from random import shuffle
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath

#!pip install spacy-udpipe
#!pip install nltk

class Enkel_Data_Augmentation():
    def __init__(self):
        self.df = pd.read_csv('synonyms.csv')
        self.df = self.df.rename(columns={'Synonym_4': 'Acutal_word'})
        self.df.drop('Unnamed: 0', axis=1, inplace=True)
        spacy_udpipe.download("sv")
        self.nlp = spacy_udpipe.load("sv")
        nltk.download('stopwords')
        self.stop_words_ = set(nltk.corpus.stopwords.words('swedish'))




    def find_closet_match(self,test_str, list2check):
     """
     This method return a word that is closest to the other words in a list.
     This is done by checking each character

     Shamelessly stolen from stackoverflow

     @param list2check: list of word
     @param test_str : the word itself
     """
     scores = {}
     for ii in list2check:
      cnt = 0
      if len(test_str) <= len(ii):
       str1, str2 = test_str, ii
      else:
       str1, str2 = ii, test_str
      for jj in range(len(str1)):
       cnt += 1 if str1[jj] == str2[jj] else 0
      scores[ii] = cnt
     scores_values = np.array(list(scores.values()))
     closest_match_idx = np.argsort(scores_values, axis=0, kind='quicksort')[-1]
     closest_match = np.array(list(scores.keys()))[closest_match_idx]
     return closest_match, closest_match_idx

    def synonyms_cadidates(self,word, df):

     doc = self.nlp(word)

     lemmatized_Word = None

     # 1. Lemmatize_word
     for token in doc:
      # print("______")
       lemmatized_Word = token.lemma_
       print("lematized :", lemmatized_Word)

     # df should be defined here

     # Find all the cadidates

     find_candidates = None

     for column in df.columns:

      try:

       find_candidates = df.loc[df[column].str.contains(lemmatized_Word, case=False)]
       #print("candidate size:", find_candidates.shape)

      except:

       pass

     if find_candidates is None:
      return False
     elif find_candidates.shape[0] == 0:
      return False

     # flat_list = [item for sublist in t for item in sublist]
     #FLatten the list of candidates
     flat_list = []
     for sublist in find_candidates.values:
      for item in sublist:
       flat_list.append(item)

     def clean_word(strink):
      newstring = ''.join([i for i in strink if not i.isdigit()])
      return newstring

     # get all the values
     # Find the closest word. that is not exactly the same word
     # print(find_closet_match(lemmatized_Word,flat_list))

     flag = True

     candidate = None

     while flag:

      cadidate, word_idx = self.find_closet_match(lemmatized_Word, flat_list)

      dot_free_candidate = cadidate.replace(".", "")

      num_free_candidate = clean_word(dot_free_candidate)

      if num_free_candidate == lemmatized_Word:
       flat_list.pop(word_idx)

      else:

       flag = False
       candidate = num_free_candidate

     return candidate

    def synonym_replacement(self,words, n, word_vec=False):
        """
        Function that replaces words with synonyms.
        :param words:
        :param n: number of synonyms to be replaced in a sentence
        :param word_vec: if True, Word2Vec model is used to replace words with synonyms.
        :return: String with replaced words. or [ False ] if no synonyms are found.
        if no synonym is found, the original sentence is returned.
        """
        word_pair_replacement = []  # list of tuples (prev_word, suggested_word)

        # Word has to be a string
        kk = words.split(" ")  # sentence

        # Filter using stopwords (nltk stopwords)
        filtered_word_list = list(set([word for word in kk if word not in self.stop_words_]))
        print(filtered_word_list)

        # choose words at random
        len_list = len(filtered_word_list)
        chosen_words = random.sample(filtered_word_list, n)

        # choose words at random
        len_list = len(filtered_word_list)
        chosen_words = random.sample(filtered_word_list, n)

        for cw in chosen_words:  # For all the chosen words, find replacements

         idx = chosen_words.index(cw)

         syn = self.synonyms_cadidates(cw, self.df)

         if syn is False:
          continue
         else:
          word_pair_replacement.append((idx, cw, syn))

        if len(word_pair_replacement) == 0:
         print("No synonym found for any words")
         return False

        for item in word_pair_replacement:
            idx = item[0]
            curr = item[1]
            synon = item[2]

            kk[idx] = synon

        listToStr = ' '.join([str(elem) for elem in kk])

        return listToStr

        return None

    def synonym_replacement_vanilla(self,words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words_]))
        random.shuffle(random_word_list)
        num_replaced = 0

        for random_word in random_word_list:
            synonym = self.synonyms_cadidates(random_word, self.df)
            if synonym is not False:
                new_words = [synonym if word == random_word else word for word in new_words]
                # what does that mean?
                # for all the words in the new_words(orignial list of sentence words)
                # if the word is equal to the random word, replace it with the synonyn
                # else, keep the word as it is
                num_replaced += 1
            if num_replaced >= n:
                break

        # this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(" ")

        return new_words



        return new_words

    def random_deletion(self, words, p):
        """
        Randomly delete words from a sentence with probability p
        :param words:
        :param p:
        :return:
        """
        #obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        #randomly delete words with probability p
        new_words = []

        for word in words:
            r = random.uniform(0, 1) # random number between 0.0 and 1.0
            if r > p: #kinda elegant when you think about it
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return new_words

    #def random_insertion(self, words, p):
    def random_insertion(self, words, n):
        """
        Randomly insert words into a sentence with probability p
        :param words:
        :param p:
        :return:
        """
        new_words = words.copy()
        for _ in range(n):
                self.add_word(new_words)

        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0

        while len(synonyms) <1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = self.synonyms_cadidates(random_word, self.df)
            counter += 1
            if counter > 10:
                return

        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)



    def enkel_augmentation(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=9):

        """
        @param sentence
        @param alpha_sr synonym replacement rate, percentage of the total sentence
        @param alpha_ri random insertion rate, percentage of the total sentence
        @param alpha_rs random swap rate, percentage of the total sentence
        @param alpha_rd random deletion rate, percentage of the total sentence
        @param num_aug how many augmented sentences to create

        inspired from : https://github.com/jasonwei20/eda_nlp/blob/04ab29c5b18d2d72f9fa5b304322aaf4793acea0/code/eda.py#L33
y

        @return list of augmented sentences
        """
        words_list = sentence.split(' ')  # list of words in the sentence
        words = [word for word in words_list if word is not '']  # remove empty words
        num_words = len(words_list)  # number of words in the sentence

        augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1 # number of augmented sentences per technique

        #synonmym replacement
        if (alpha_sr > 0):
                n_sr = max(1, int(alpha_sr * num_words)) # number of words to be replaced per technique
                print("Number of words to be replaced per technique: ", n_sr)
                for _ in range(num_new_per_technique):
                    a_words = self.synonym_replacement_vanilla(words, n_sr)
                    augmented_sentences.append(' '.join(a_words))

        #random insertion
        if (alpha_ri > 0):
            n_ri = max(1,int(alpha_ri * num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))


        #Random Deletion
        if (alpha_rd > 0):
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(words, alpha_rd)
                print(a_words)
                augmented_sentences.append(' '.join(a_words))






        return augmented_sentences


