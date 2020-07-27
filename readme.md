# Partial Data LDA for Levantine Arabic

Data and scripts used for a thesis paper on [Targeted Topic Modeling for Levantine Arabic](http://uu.diva-portal.org/smash/record.jsf?pid=diva2%3A1439483&dswid=-4687). 

## Getting Started

In this repo, you will find: 
1. data of raw text tweets in mixed varieties of Arabic (including MSA) as well as automatically filtered Levantine Arabic tweets. 
2. A script to generate topic models from partial data using LDA. 
3. A pretrained model to classify Levantine Arabic texts

To get a copy of the code, use the git clone command. 

```shell
git clone git@github.com:snekz/lev_thesis.git
```

### Prerequisites:

Make sure you have these modules installed for testing LDA-pd and/or for using the pretrained Levantine Arabic filtering model. 

### For Partial Data LDA: 

```shell
pip install pandas
pip install numpy
pip install gensim 
pip install tabulate
pip install pprint
```
### For Dialect Filtering

```shell
pip install json
pip install keras
pip install numpy 
```

## Twitter Data

### mixed_tweets.txt
The data consists of 51986 Tweets of mixed varieites of Arabic (including MSA) collected between January and May 2020. All tweets contain some mention of Covid19 complete with hashtags and @ mentions. They contain no images/video or links. Information on users, such as usernames or demographics, are not included. Tweets were scraped based on geographical location (Levant area) and containment of selected Levantine Arabic stopwords. 

### filtered_preprocessed_tweets.txt

Up to cleaned 11399 tweets in Levantine Arabic (automatically filtered, see "Dialect Filtering"). Preprocessing is minimal with no stemming. For example, prefix 'و' (the 'and' conjunct in Arabic) is still attached to many words. File "mixed_lev_noCovid.txt" contains tweets with all mentions of covid19 (in their different formats) removed. 

## Examples

### Dialect Filtering
A large number of linguistic varieties exist in Arab countries. These can be categorized by region (or even by city for more fine-grained analysis), ethnicity, social status, and proximity to the city. 

Experiments described in the paper filter out all tweets written in MSA or any linguistic variety other than Levantine Arabic. In the paper, I used a deep learning model (originally developed by M. Elaraby and M. Abdul-Mageed for the AOC repository [[3]](#3)) and modified it to perform coarse-grained binary classification to single out Levantine Arabic tweets. 

I am including the model used in the paper ```dialect_identification.ipynb```, which was pre-trained on a combination of data form AOC (provided here https://github.com/UBC-NLP/aoc_id) and from Subtask 1 of The MADAR Shared Task on Arabic Fine-Grained Dialect Identification [[2]](#2).

### Partial LDA
If you want to run the example described in the paper, you need to load the Levantine Arabic Multidialectal Word Embeddings [[1]](#1). 

1. Since these emebddings are binary files, use [Word2tensor](https://github.com/HichemMaiza/Word2tensor) to convert them into word2vec TSV format. 

```shell
word2vec2tensor.py -i lev.bin -o lev.tensor -b
```

2. Because these embeddings have Arabic words transliterated in a fashion similar to Buckwalter style, it's much more convenient to convert them into their original Arabic alphabet. You can do this using ```maper.py``` in this format:

```shell
python mapper.py lev.tensor lev_ar.tensor
```

Feel free to play around with parameters in ```lda_pd.py```.

## Credit & Acknowledgments

* **Arabic Multidialectal Word Embeddings** for providing multidialect word emebddings
Copyright 2018 of New York University Abu Dhabi. 

Download here: https://camel.abudhabi.nyu.edu/arabic-multidialectal-embeddings/

<a id="1">[1]</a> 
Erdmann, A., Zalmout, N., & Habash, N. (2018, July). 
Addressing noise in multidialectal word embeddings.  
In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 558-565).

* **The MADAR Shared Task on Arabic Fine-Grained Dialect Identification** for providing training data of different Arabic dialects

Read here: https://www.aclweb.org/anthology/W19-4622

<a id="2">[2]</a> 
Bouamor, H., Hassan, S., & Habash, N. (2019, August).
The MADAR shared task on Arabic fine-grained dialect identification. 
In Proceedings of the Fourth Arabic Natural Language Processing Workshop (pp. 199-207).

* **Deep Models for Arabic Dialect Indentification on Benchmarked Data** for providing deep learning models to use with Arabic texts

Read here: https://www.aclweb.org/anthology/W18-3930
Git repo: https://github.com/UBC-NLP/aoc_id

<a id="3">[3]</a> 
Elaraby, M., & Abdul-Mageed, M. (2018, August).
Deep models for arabic dialect identification on benchmarked data. 
In Proceedings of the Fifth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2018) (pp. 263-274).

' **Hisham Maiza** for his [Word2Tensor](https://github.com/HichemMaiza/Word2tensor) script. 
