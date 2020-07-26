## Requirements:

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

## Twitter Database

### mixed_tweets.txt
The data consists of 51986 Tweets of mixed varieites of Arabic (including MSA) collected between January and May 2020. All tweets contain some mention of Covid19 complete with hashtags and @ mentions. They contain no images/video or links. Information on users, such as usernames or demographics, are not included. Tweets were scraped based on geographical location (Levant area) and containment of selected Levantine Arabic stopwords. 

### filtered_preprocessed_tweets.txt

Up to cleaned 11399 tweets in Levantine Arabic (automatically filtered, see "Dialect Filtering"). Preprocessing is minimal with no stemming. For example, prefix 'و' (the 'and' conjunct in Arabic) is still attached to many words. File "mixed_lev_noCovid.txt" contains tweets with all mentions of covid19 (in their different formats) removed. 

## Examples

### Dialect Filtering
A large number of linguistic varieties exist in Arab countries. These can be categorized by region (or even by city for more fine-grained analysis), ethnicity, social status, and proximity to the city. 

Experiments described in the paper filter out all tweets written in MSA or any linguistic variety other than Levantine Arabic. In the paper, I used a deep learning model (originally developed by M. Elaraby and M. Abdul-Mageed for the AOC repository, read more here: https://www.aclweb.org/anthology/W18-3930/) and modified it to perform coarse-grained binary classification to single out Levantine Arabic tweets. 

I am including the model used in the paper, which was pre-trained on a combination of data form AOC (provided here https://github.com/UBC-NLP/aoc_id) and from Subtask 1 of The MADAR Shared Task on Arabic Fine-Grained Dialect Identification (https://www.aclweb.org/anthology/W19-4622/). 


### Partial LDA
If you want to run the example described in the paper, you need to load the Levantine Arabic Multidialectal Word Embeddings (which you can get from https://camel.abudhabi.nyu.edu/arabic-multidialectal-embeddings/).

1. Since these emebddings are binary files, use Word2tensor (https://github.com/HichemMaiza/Word2tensor) to convert them into word2vec TSV format. 

```shell
word2vec2tensor.py -i lev.bin -o lev.tensor -b
```

2. Because these embeddings have Arabic words transliterated in a fashion similar to Buckwalter style, it's much more convenient to convert them into their original Arabic alphabet. You can do this using maper.py in this format:

```shell
python mapper.py lev.tensor lev_ar.tensor
```


## Tools & License

### Arabic Multidialectal Word Embeddings 
Copyright 2018 of New York University Abu Dhabi. 

Download here: https://camel.abudhabi.nyu.edu/arabic-multidialectal-embeddings/

@inproceedings{erdmann-etal-2018-addressing,
    title = "Addressing Noise in Multidialectal Word Embeddings",
    author = "Erdmann, Alexander  and
      Zalmout, Nasser  and
      Habash, Nizar",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-2089",
    doi = "10.18653/v1/P18-2089",
    pages = "558--565",
    abstract = "Word embeddings are crucial to many natural language processing tasks. The quality of embeddings relies on large non-noisy corpora. Arabic dialects lack large corpora and are noisy, being linguistically disparate with no standardized spelling. We make three contributions to address this noise. First, we describe simple but effective adaptations to word embedding tools to maximize the informative content leveraged in each training sentence. Second, we analyze methods for representing disparate dialects in one embedding space, either by mapping individual dialects into a shared space or learning a joint model of all dialects. Finally, we evaluate via dictionary induction, showing that two metrics not typically reported in the task enable us to analyze our contributions{'} effects on low and high frequency words. In addition to boosting performance between 2-53{\%}, we specifically improve on noisy, low frequency forms without compromising accuracy on high frequency forms.",
}

### Deep Moelfs for Arabic Dialect Indentification on Benchmarked Data

@inproceedings{elaraby-abdul-mageed-2018-deep,
    title = "Deep Models for {A}rabic Dialect Identification on Benchmarked Data",
    author = "Elaraby, Mohamed  and
      Abdul-Mageed, Muhammad",
    booktitle = "Proceedings of the Fifth Workshop on {NLP} for Similar Languages, Varieties and Dialects ({V}ar{D}ial 2018)",
    month = aug,
    year = "2018",
    address = "Santa Fe, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-3930",
    pages = "263--274",
    abstract = "The Arabic Online Commentary (AOC) (Zaidan and Callison-Burch, 2011) is a large-scale repos-itory of Arabic dialects with manual labels for4varieties of the language. Existing dialect iden-tification models exploiting the dataset pre-date the recent boost deep learning brought to NLPand hence the data are not benchmarked for use with deep learning, nor is it clear how much neural networks can help tease the categories in the data apart. We treat these two limitations:We (1) benchmark the data, and (2) empirically test6different deep learning methods on thetask, comparing peformance to several classical machine learning models under different condi-tions (i.e., both binary and multi-way classification). Our experimental results show that variantsof (attention-based) bidirectional recurrent neural networks achieve best accuracy (acc) on thetask, significantly outperforming all competitive baselines. On blind test data, our models reach87.65{\%}acc on the binary task (MSA vs. dialects),87.4{\%}acc on the 3-way dialect task (Egyptianvs. Gulf vs. Levantine), and82.45{\%}acc on the 4-way variants task (MSA vs. Egyptian vs. Gulfvs. Levantine). We release our benchmark for future work on the dataset",
}


