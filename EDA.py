import pandas as pd
import numpy as np
import spacy_udpipe
import nltk
import random

#!pip install spacy-udpipe
#!pip install nltk


stop_words=['aderton',
 'adertonde',
 'adjö',
 'aldrig',
 'alla',
 'allas',
 'allt',
 'alltid',
 'alltså',
 'andra',
 'andras',
 'annan',
 'annat',
 'arton',
 'artonde',
 'att',
 'av',
 'bakom',
 'bara',
 'behöva',
 'behövas',
 'behövde',
 'behövt',
 'beslut',
 'beslutat',
 'beslutit',
 'bland',
 'blev',
 'bli',
 'blir',
 'blivit',
 'bort',
 'borta',
 'bra',
 'bäst',
 'bättre',
 'båda',
 'bådas',
 'dag',
 'dagar',
 'dagarna',
 'dagen',
 'de',
 'del',
 'delen',
 'dem',
 'den',
 'deras',
 'dess',
 'det',
 'detta',
 'dig',
 'din',
 'dina',
 'dit',
 'ditt',
 'dock',
 'du',
 'där',
 'därför',
 'då',
 'efter',
 'eftersom',
 'elfte',
 'eller',
 'elva',
 'en',
 'enkel',
 'enkelt',
 'enkla',
 'enligt',
 'er',
 'era',
 'ert',
 'ett',
 'ettusen',
 'fanns',
 'fem',
 'femte',
 'femtio',
 'femtionde',
 'femton',
 'femtonde',
 'fick',
 'fin',
 'finnas',
 'finns',
 'fjorton',
 'fjortonde',
 'fjärde',
 'fler',
 'flera',
 'flesta',
 'fram',
 'framför',
 'från',
 'fyra',
 'fyrtio',
 'fyrtionde',
 'få',
 'får',
 'fått',
 'följande',
 'för',
 'före',
 'förlåt',
 'förra',
 'första',
 'genast',
 'genom',
 'gick',
 'gjorde',
 'gjort',
 'god',
 'goda',
 'godare',
 'godast',
 'gott',
 'gälla',
 'gäller',
 'gällt',
 'gärna',
 'gå',
 'går',
 'gått',
 'gör',
 'göra',
 'ha',
 'hade',
 'haft',
 'han',
 'hans',
 'har',
 'heller',
 'hellre',
 'helst',
 'helt',
 'henne',
 'hennes',
 'hit',
 'hon',
 'honom',
 'hundra',
 'hundraen',
 'hundraett',
 'hur',
 'här',
 'hög',
 'höger',
 'högre',
 'högst',
 'i',
 'ibland',
 'idag',
 'igen',
 'igår',
 'imorgon',
 'in',
 'inför',
 'inga',
 'ingen',
 'ingenting',
 'inget',
 'innan',
 'inne',
 'inom',
 'inte',
 'inuti',
 'ja',
 'jag',
 'jämfört',
 'kan',
 'kanske',
 'knappast',
 'kom',
 'komma',
 'kommer',
 'kommit',
 'kr',
 'kunde',
 'kunna',
 'kunnat',
 'kvar',
 'legat',
 'ligga',
 'ligger',
 'lika',
 'likställd',
 'likställda',
 'lilla',
 'lite',
 'liten',
 'litet',
 'länge',
 'längre',
 'längst',
 'lätt',
 'lättare',
 'lättast',
 'långsam',
 'långsammare',
 'långsammast',
 'långsamt',
 'långt',
 'man',
 'med',
 'mellan',
 'men',
 'mer',
 'mera',
 'mest',
 'mig',
 'min',
 'mina',
 'mindre',
 'minst',
 'mitt',
 'mittemot',
 'mot',
 'mycket',
 'många',
 'måste',
 'möjlig',
 'möjligen',
 'möjligt',
 'möjligtvis',
 'ned',
 'nederst',
 'nedersta',
 'nedre',
 'nej',
 'ner',
 'ni',
 'nio',
 'nionde',
 'nittio',
 'nittionde',
 'nitton',
 'nittonde',
 'nog',
 'noll',
 'nr',
 'nu',
 'nummer',
 'när',
 'nästa',
 'någon',
 'någonting',
 'något',
 'några',
 'nödvändig',
 'nödvändiga',
 'nödvändigt',
 'nödvändigtvis',
 'och',
 'också',
 'ofta',
 'oftast',
 'olika',
 'olikt',
 'om',
 'oss',
 'på',
 'rakt',
 'redan',
 'rätt',
 'sade',
 'sagt',
 'samma',
 'sedan',
 'senare',
 'senast',
 'sent',
 'sex',
 'sextio',
 'sextionde',
 'sexton',
 'sextonde',
 'sig',
 'sin',
 'sina',
 'sist',
 'sista',
 'siste',
 'sitt',
 'sju',
 'sjunde',
 'sjuttio',
 'sjuttionde',
 'sjutton',
 'sjuttonde',
 'sjätte',
 'ska',
 'skall',
 'skulle',
 'slutligen',
 'små',
 'smått',
 'snart',
 'som',
 'stor',
 'stora',
 'stort',
 'större',
 'störst',
 'säga',
 'säger',
 'sämre',
 'sämst',
 'så',
 'tack',
 'tidig',
 'tidigare',
 'tidigast',
 'tidigt',
 'till',
 'tills',
 'tillsammans',
 'tio',
 'tionde',
 'tjugo',
 'tjugoen',
 'tjugoett',
 'tjugonde',
 'tjugotre',
 'tjugotvå',
 'tjungo',
 'tolfte',
 'tolv',
 'tre',
 'tredje',
 'trettio',
 'trettionde',
 'tretton',
 'trettonde',
 'två',
 'tvåhundra',
 'under',
 'upp',
 'ur',
 'ursäkt',
 'ut',
 'utan',
 'utanför',
 'ute',
 'vad',
 'var',
 'vara',
 'varför',
 'varifrån',
 'varit',
 'varken',
 'varsågod',
 'vart',
 'vem',
 'vems',
 'verkligen',
 'vi',
 'vid',
 'vidare',
 'viktig',
 'viktigare',
 'viktigast',
 'viktigt',
 'vilka',
 'vilken',
 'vilket',
 'vill',
 'vänster',
 'vänstra',
 'värre',
 'vår',
 'våra',
 'vårt',
 'än',
 'ännu',
 'även',
 'åtminstone',
 'åtta',
 'åttio',
 'åttionde',
 'åttonde',
 'över',
 'övermorgon',
 'överst',
 'övre']

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

     # 1. Lemmatize_word
     for token in doc:
      # print("Word Token:",token)
      # print("Lemmatized Token:",token.lemma_)
      # print("Token POS TAG:",token.pos_)

      # print("______")
      lemmatized_Word = token.lemma_
      print("lematized :", lemmatized_Word)

      # lemmatized_Word = word

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
