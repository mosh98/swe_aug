# EDA: Easy Data Augmentation in Swedish
## or Enkel Data Augmentation


## What is EDA?
A way to augment data in a way that is easy to understand and use. There are 4 mains components
1. Random Synomym Replacement
2. Random Word Replacement
3. Random Word Deletion
4. Random Word Insertion

``!git clone https://github.com/mosh98/swe_aug.git``

This is built on top of a swedish word2vec. Make sure you download that first.


````python
!wget https://www.ida.liu.se/divisions/hcs/nlplab/swectors/swectors-300dim.txt.bz2
!bzip2 -dk /content/swectors-300dim.txt.bz2
!pip install -r reqs.txt
word_vec_path = '/content/swectors-300dim.txt' #path to txt vector file
````


### import the library 
```python
from swe_aug import EDA
aug = EDA.Enkel_Data_Augmentation(word_vec_path)

aug.enkel_augmentation(item.item(),alpha_sr=0.1, alpha_ri=0.3, alpha_rs=0.2, alpha_rd=0.1, num_aug=4)
```


Thats it buddy!
