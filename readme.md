# Swedish Augmentation Package

Includes many different Augmentation packages for Swedish.



----------------------------------------------------
### EDA: Easy Data Augmentation in Swedish
### What is EDA?
A way to augment data in a way that is easy to understand and use. There are 4 mains components
1. Random Synomym Replacement
2. Random Word Replacement
3. Random Word Deletion
4. Random Word Insertion

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

### Step 3
#### Import EDA:

____________________________________________________________________

### EDA [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/118UDmQzHtO3UmO0HroL4nthNG4Qea-k8?usp=sharing)


```python
from swe_aug import EDA
aug = EDA.Enkel_Data_Augmentation(word_vec_path)

txt = "Killen planerade att resa till Kurdistan med sin pappa och sin mamma. "
```

````python

augmented_sentences = aug.enkel_augmentation(txt, alpha_sr=0.1, 
                                             alpha_ri=0.3, alpha_rs=0.2, 
                                             alpha_rd=0.1, num_aug=4)
#returns a list of augmented sentences
````
___________________________________________________________________________



#Text Fragmenter ![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)
```python
from swe_aug.Other_Techniques import Text_Cropping

frag = Text_Cropping.cropper(percent = 0.25)
list_of_fragmented_sentence = frag.text_fragmeter(txt)
# chops sentence into 4 halfs.
```


Thats it buddy!

More Comming!

#### References
[1] Swedish word2vec: https://www.ida.liu.se/divisions/hcs/nlplab/swectors/

[2] EDA: https://aclanthology.org/D19-1670/

[3] Text Fragmenter: That was me
