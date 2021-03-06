# Swedish Augmentation Packages

Includes many different Augmentation packages for Swedish.



----------------------------------------------------
## How do i setup?

### Step 1
``!git clone https://github.com/mosh98/swe_aug.git``

This is built on top of a swedish word2vec. Make sure you download that first.

### Step 2
````python
!wget https://www.ida.liu.se/divisions/hcs/nlplab/swectors/swectors-300dim.txt.bz2
!bzip2 -dk /content/swectors-300dim.txt.bz2
!pip install -r reqs.txt


word_vec_path = '/content/swectors-300dim.txt' #path to txt vector file

#you can even set path to your own pretrain word2vec (make sure its a txt file)
````


#### Then Use your desired augmentation package


____________________________________________________________________

### EDA [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/118UDmQzHtO3UmO0HroL4nthNG4Qea-k8?usp=sharing)

### EDA: Easy Data Augmentation in Swedish
### What is EDA? [2]
A way to augment data in a way that is easy to understand and use. There are 4 mains components
1. Random Synomym Replacement
2. Random Word Replacement
3. Random Word Deletion
4. Random Word Insertion


```python
from swe_aug import EDA
aug = EDA.Enkel_Data_Augmentation(word_vec_path)

txt = "enter ur desired text. It can be a sentence or a paragraph"
```

````python

augmented_sentences = aug.enkel_augmentation(txt, alpha_sr=0.1, 
                                             alpha_ri=0.3, alpha_rs=0.2, 
                                             alpha_rd=0.1, num_aug=4)
#returns a list of augmented sentences
````

___________________________________________________________________________



# Text Fragmenter ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)
```python
from swe_aug.Other_Techniques import Text_Cropping

frag = Text_Cropping.cropper(percent = 0.25)
list_of_fragmented_sentence = frag.text_fragmeter(txt)
# chops sentence into 4 halfs.
```




##  Type Specific Similar word Replacement ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)
Idea is to replace word that are similar in an embeddings space that has the same POS token. [4]
````python
# "NOUN", "VERB", "ADJ", "ADV", "PROPN","CONJ"
#These are the tokens you can perturb! [CASE SENSITIVE!]

from swe_aug.Other_Techniques import Type_SR
aug = Type_SR.type_DA(word_vec_path)

list_of_augs = aug.type_synonym_sr(txt, token_type = "NOUN", n = 2)



````

#### References
[1] Swedish word2vec: https://www.ida.liu.se/divisions/hcs/nlplab/swectors/

[2] EDA: https://aclanthology.org/D19-1670/

[3] Text Fragmenter: That was me

[4] Type Specific: That was me too

### Cite?
````
@software{Mahamud2022,
  author = {Mahamud,Mosleh},
  title = {Swedish Augmentation Packages},
  year = {2022},
  publisher = {GitHub},
  journal = {Not Decided yet},
  howpublished = {\url{https://github.com/mosh98/swe_aug}},
}
````
