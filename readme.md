# Partial Data LDA for Levantine Arabic

Data and scripts used for a thesis paper on [Targeted Topic Modeling for Levantine Arabic](http://uu.diva-portal.org/smash/record.jsf?pid=diva2%3A1439483&dswid=-4687). 

## Getting Started

To get a copy of the code, use the git clone command. 

```shell
git clone git@github.com:snekz/lev_lda.git
```
### Contents

In this repo, you will find: 

1. data of raw text tweets in mixed varieties of Arabic (including MSA) as well as automatically filtered Levantine Arabic tweets. 

2. A script to generate topic models from partial data using LDA. The data included is determined by whether a document contains one or more keywords from a list of pre-selected terms (which can be supplemented by similar terms from word embeddings).

3. A pretrained model to classify Levantine Arabic texts

### Prerequisites:

Make sure you have these modules installed for testing LDA-pd. Those required for dialect identification can be found in ```dialect_identification.ipynb``` when executed in google colab. 

```shell
pip install pandas
pip install numpy
pip install gensim 
pip install tabulate
pip install pprint
```

## Twitter Data

### Mixed varities 

The ```mixed_tweets.txt``` data collection consists of 51986 Tweets of mixed varieites of Arabic (including MSA) collected between January and May 2020. All tweets contain some mention of **Covid19** complete with hashtags and @ mentions. They contain no images/video or links. Information on users, such as usernames or demographics, are not included. Tweets were scraped based on geographical location (Levant area) and containment of selected Levantine Arabic stopwords. 

### filtered_preprocessed_tweets.txt

File ```mixed_lev.txt``` contains up to cleaned 11399 tweets written in Levantine Arabic (automatically filtered, see "Dialect Filtering" example). Preprocessing is minimal with no stemming. For example, prefix 'و' (the 'and' conjunct in Arabic) is still attached to many words. File ```mixed_lev_noCovid.txt``` contains tweets with all mentions of covid19 (in their different formats) removed. 

## Examples

### Dialect Filtering
A large number of linguistic varieties exist in Arab countries. These can be categorized by region (or even by city for more fine-grained analysis), ethnicity, social status, and proximity to the city. 

Experiments described in the paper filter out all tweets written in MSA or any linguistic variety other than Levantine Arabic. In the paper, I used a deep learning model (originally developed by M. Elaraby and M. Abdul-Mageed for the AOC repository [[1]](#1)) and modified it to perform coarse-grained binary classification to single out Levantine Arabic tweets. 

I am including the model used in the paper ```dialect_identification.ipynb```, which was pre-trained on a combination of data form AOC (provided here https://github.com/UBC-NLP/aoc_id) and from Subtask 1 of The MADAR Shared Task on Arabic Fine-Grained Dialect Identification [[2]](#2).

### Partial LDA
If you want to run the example described in the paper, you need to download and load the [Levantine Arabic Multidialectal Word Embeddings](https://camel.abudhabi.nyu.edu/arabic-multidialectal-embeddings/) [[3]](#3). 

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

* **Deep Models for Arabic Dialect Indentification on Benchmarked Data**[[1]](#1) for providing deep learning models to use with Arabic texts. [**paper**](https://www.aclweb.org/anthology/W18-3930) | [**repo**](https://github.com/UBC-NLP/aoc_id)

* **The MADAR Shared Task on Arabic Fine-Grained Dialect Identification**[[2]](#2) for providing training data of different Arabic dialects. [**paper**](https://www.aclweb.org/anthology/W19-4622)

* **Arabic Multidialectal Word Embeddings**[[3]](#3)
(Copyright 2018 of New York University Abu Dhabi.) [**download**](https://camel.abudhabi.nyu.edu/arabic-multidialectal-embeddings/) | [**paper**](https://www.aclweb.org/anthology/P18-2089/)

* **Hisham Maiza** for his [Word2Tensor](https://github.com/HichemMaiza/Word2tensor) script. 

* **TwitterScraper** for their efficient [twitter scraping script](https://github.com/taspinar/twitterscraper). 

## References

<a id="1">[1]</a> 
Elaraby, M., & Abdul-Mageed, M. (2018, August).
Deep models for arabic dialect identification on benchmarked data. 
In Proceedings of the Fifth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2018) (pp. 263-274).

<a id="2">[2]</a> 
Bouamor, H., Hassan, S., & Habash, N. (2019, August).
The MADAR shared task on Arabic fine-grained dialect identification. 
In Proceedings of the Fourth Arabic Natural Language Processing Workshop (pp. 199-207).

<a id="3">[3]</a> 
Erdmann, A., Zalmout, N., & Habash, N. (2018, July). 
Addressing noise in multidialectal word embeddings.  
In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 558-565).
