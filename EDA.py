import pandas as pd
import numpy as np
import spacy_udpipe

import nltk
import random

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
       print("candidate size:", find_candidates.shape)

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

    def enkel_augmentation(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=9):

        """
        @param sentence
        @param alpha_sr synonym replacement rate, percentage of the total sentence
        @param alpha_ri random insertion rate, percentage of the total sentence
        @param alpha_rs random swap rate, percentage of the total sentence
        @param alpha_rd random deletion rate, percentage of the total sentence
        @param num_aug how many augmented sentences to create

        inspired from : https://github.com/jasonwei20/eda_nlp/blob/04ab29c5b18d2d72f9fa5b304322aaf4793acea0/code/eda.py#L33

        @return list of augmented sentences
        """
        words = sentence.split(' ')  # list of words in the sentence
        words = [word for word in words if word is not '']  # remove empty words
        num_words = len(words)  # number of words in the sentence

