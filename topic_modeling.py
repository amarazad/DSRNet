#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
import pandas as pd
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


import glob, os

path = './kummerfeld/Corpus_processed/combined_disentangle_data/train'                     # use your path
all_files = glob.glob(os.path.join(path, "*"))

df_from_each_file = (pd.read_csv(f, sep="\t") for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)


def get_key(my_dict, val, doc_id):
    for key, value in my_dict.items(): 
         if val in value and key.split('_')[0] == doc_id: 
            return key 
  
    return "key doesn't exist"

#disentanglement
from tqdm import tqdm

concatenated_df = concatenated_df.assign(Conv_Id=pd.Series(np.random.randn(len(concatenated_df['time']))).values)
    
english_corpus = ['train']
corpus_path = './kummerfeld/data'

for en_corpus in english_corpus:
    docs_path = os.path.join(corpus_path, en_corpus)
    docs = os.listdir(docs_path)
    
    
    conversation = {}
    conv_id = 0
    conversation[str(1) + '_'+ str(conv_id)] = []
    i = 0
    
    for doc in tqdm(docs):
        visited = []
                
        if doc.split('.')[-1] == 'txt' and doc.split('.')[-2] == 'annotation':
            doc_path = os.path.join(docs_path, doc)
            i += 1
            conversation[str(i)+'_'+str(conv_id)] = []
            
            with open(doc_path) as fin:
                for line in fin.readlines():
                    line = line.strip()
                    
                    prev_talk = line.split()[0]
                    next_talk = line.split()[1]
                        
                    if prev_talk not in conversation[str(i)+'_'+str(conv_id)]:
                        if prev_talk  not in visited:
                            conv_id += 1
                            if prev_talk == next_talk:
                                conversation[str(i)+'_'+str(conv_id)] = [next_talk]
                            else:
                                conversation[str(i)+'_'+str(conv_id)] = [prev_talk, next_talk]
                        else:
                            conversation[get_key(conversation,str(prev_talk), str(i))].append(next_talk)                            
                    else:
                        conversation[str(i)+'_'+str(conv_id)].append(next_talk)
                        
                    visited.append(prev_talk)
                    visited.append(next_talk)
                        
#creating reverse map of line num to conv id
rev_conversation = {}
for k, v in conversation.items():
    doc_num = k.split('_')[0]
    conv_id = k.split('_')[1]
    if  doc_num not in rev_conversation:
        rev_conversation[doc_num] = {}
    for line in v:
        rev_conversation[doc_num][line] = conv_id

#final dataframe with conversations segregated
for index, row in concatenated_df.iterrows():
    if row['Id'].split('_')[1] in rev_conversation and row['Id'].split('_')[2] in rev_conversation[row['Id'].split('_')[1]]:
        concatenated_df.at[index,'Conv_Id'] = rev_conversation[row['Id'].split('_')[1]][row['Id'].split('_')[2]] 
    else:
        concatenated_df.at[index,'Conv_Id'] = 0
        

#finding and dropping rows with conv id = 0..i.e. the context utterances 
indexNames = concatenated_df[concatenated_df['Conv_Id'] == 0 ].index
 
# Delete these row indexes from dataFrame
concatenated_df.drop(indexNames , inplace=True)

#grouping utterances based on Conv Id
grouped = concatenated_df.groupby('Conv_Id')
uttr = {}
for name, group in grouped:
    uttr[name] = list(group['Utterance'])

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

uttr = {k :  ' '.join([' '.join([word for word in word_tokenize(sent) if word != 'COMMAND' and word != 'FILEPATH' and word != 'EMOJI' and word != 'URL' and 'EMAIL' not in word]) for sent in v]) for k, v in uttr.items()}
corpus = [string for string in uttr.values()]
vectorizer = TfidfVectorizer(min_df = 0, stop_words = 'english', tokenizer = word_tokenize ,sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()

x =   [string for string in uttr.values()]
word_count = []
for string in x:
    for w in word_tokenize(string):
        word_count.append(w)


#removing words with less and very high tf-idf scores, keeping words with scores between 0.3 to 0.5
#first level filter with tf-idf scores
import nltk
words = set(nltk.corpus.words.words())
from nltk.stem.porter import *
stemmer = PorterStemmer()
uttr_stemmed = {}

count = 0
for key, value in uttr.items():
    words_to_keep = ''
    doc = count
    feature_index = tfidf_matrix[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        
        if s >= 0.3 and s <= 0.55:
            words_to_keep = words_to_keep + ' ' + stemmer.stem(w)
                
    uttr_stemmed[key] = words_to_keep
    count += 1
    
print('first round of tf-idf filtering')

#second level filter with idf values
#removing words with less and very low idf scores, keeping words with scores above 0.5

corpus = [string for string in uttr_stemmed.values()]
vectorizer = TfidfVectorizer(stop_words = 'english', tokenizer = word_tokenize ,use_idf=True)
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()

idf_dict = {name : vectorizer.idf_[i] for i, name in enumerate(feature_names)}

uttr_second_filter = {}

count = 0
for key, value in uttr_stemmed.items():
    words_to_keep = ''
    doc = count
    for w in word_tokenize(value):
        try:
            s = idf_dict[w]
        except:
            continue
        if s <= 7.5:
            words_to_keep = words_to_keep + ' ' + w
                
    uttr_second_filter[key] = words_to_keep
    count += 1
    
#uttr_second_filter
print('second round of idf filtering')

#for conv level
data = list(uttr_second_filter.values())
data = [doc[:100000] if len(doc) >= 100000 else doc for doc in data] #because 1000000 is the max limit of tokens for spacy
utters_refined = []

for u in data:
    sent = ' '.join([word for word in u.split() if word != 'COMMAND' and word != 'FILEPATH' and word != 'EMOJI' and word != 'URL' and 'EMAIL' not in word])
    
    if len(sent.split()) >= 3:
        utters_refined.append(sent)
        

data_words = [utter.split() for utter in utters_refined]


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# Form Bigrams
data_words_bigrams = make_bigrams(data_words)

from tqdm import tqdm

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

print('Loading done')
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


mallet_path = 'mallet-2.0.8/bin/mallet' # update this path #num_of_topics = 40 -> optimum for utterance level
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=80, id2word=id2word, random_seed=10)
#ldamallet.save('ldamallet.model')

# final - 80
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


#finding the dominant topic of each sentence
def format_topics_sentences(ldamodel, corpus=corpus, texts=data):    
    # Get main topic in each utterance
    model_corpus = ldamodel[corpus]
    print('Loading done...')
    
    for i, row in enumerate(tqdm(model_corpus)):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                texts[i]['topics'] = topic_keywords
            else:
                break
    
    return texts

#generating topic datasets
def make_data(mode, task, add_q=False):
    with open('kummerfeld/ctxt-'+mode+task+'.txt', 'r') as f:
        lines = f.readlines()
    print(mode, len(lines))
    if add_q:
        line_s = [word_tokenize(line.split('[eoc]')[0].replace('[eos]', '').strip('\n')) for line in lines]
    else:
        line_s = [word_tokenize(line.split('[sep]')[0].replace('[eos]', '').strip('\n')) for line in lines] 
    
    #creating corpus for topic modeling
    corpus = [id2word.doc2bow(text) for text in line_s]
    print(len(corpus))
    
    #print(line)
    if not add_q : data = {i : {'context' : line.split('[sep]')[0], 'qstn': '', 'text' : line.split('[sep]')[1],  'topics' : ''} for i, line in enumerate(lines)}
    if add_q : data = {i : {'context' : line.split('[eoc]')[0], 'qstn': line.split('[eoq]')[0].split('[eoc]')[1], 'text' : line.split('[sep]')[1],  'topics' : ''} for i, line in enumerate(lines)}

    tt = format_topics_sentences(ldamodel=ldamallet, corpus=corpus, texts=data)
    print(len(tt))

    with open('kummerfeld/ctxt-'+mode+task+'-topic.txt', 'w') as f:
        for k, v in tt.items():
            if not add_q : string = v['context'].strip() + '[eoc] ' + v['topics'].strip() + ' [eot] [sep] ' + v['text'].strip().strip('\n')
            if add_q : string = v['context'].strip() + '[eoc] ' + v['qstn'].strip() + ' [eoq] ' + v['topics'].strip() + ' [eot] [sep] ' + v['text'].strip().strip('\n')
            f.write('%s\n' %string)

import argparse    
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--add_qstn", default=False, type=bool,help="Whether to add question or not")
arg = parser.parse_args()
modes = ['train', 'dev', 'test']
task = ''
if arg.add_qstn: task = '-qstnYN'
for mode in modes:
    make_data(mode, task, arg.add_qstn)
