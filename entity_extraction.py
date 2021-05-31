#!/usr/bin/env python
# coding: utf-8

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


#import data
import glob, os
# Entity extaction is done using spacy in the following. 
# For domain specfic entity extraction refer Mohapatra et. al. The code is not included here due to propriety reasons.
#from domain_entity_extraction import get_entities #inject your code for Mohapatra et. al.



path = './kummerfeld/Corpus_processed/combined_disentangle_data/train'                     # use your path
all_files = glob.glob(os.path.join(path, "*"))     # advisable to use os.path.join as this makes concatenation OS independent

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

print('grouping utterances based on Conv Id : ', len(uttr), len(grouped))


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

import spacy
nlp = spacy.load('en_core_web_sm')

def get_entities(txt):
    doc1 = nlp(txt)
    return list(doc1.ents)

#finding the dominant topic of each sentence
def format_entity_sentences(texts):
    
    for i in range(len(texts)):
        #ent, spent = get_entities(texts[i]['context']) #inject your code for Mohapatra et. al.
        ent = get_entities(texts[i]['context'])
        texts[i]['entity'] = ', '.join(ent)
    return texts

#generating topic datasets
def make_data(mode, task, add_q=False):
    with open('kummerfeld/ctxt-'+mode+task+'.txt', 'r') as f:
        lines = f.readlines()
    print(mode, len(lines))

    data = {}
    if not add_q:
        for i,line in enumerate(lines):
            line_parts = line.split('[sep]')
            data[i] = {'context' : line_parts[0], 'qstn': '', 'text' : line_parts[1],  'entity' : ''}
    if add_q : data = {i : {'context' : line.split('[eoc]')[0], 'qstn': line.split('[eoq]')[0].split('[eoc]')[1], 'text' : line.split('[sep]')[1],  'entity' : ''} for i, line in enumerate(lines)}

    tt = format_entity_sentences(texts=data)
    print(len(tt))

    with open('kummerfeld/ctxt-'+mode+task+'-entity.txt', 'w') as f:
        for k, v in tt.items():
            if not add_q : string = v['context'].strip() + '[eoc] ' + v['entity'].strip() + ' [eot] [sep] ' + v['text'].strip().strip('\n')
            if add_q : string = v['context'].strip() + '[eoc] ' + v['qstn'].strip() + ' [eoq] ' + v['entity'].strip() + ' [eot] [sep] ' + v['text'].strip().strip('\n')
            f.write('%s\n' %string)

import argparse    
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--add_qstn", default=False, type=bool,help="Whether to add question or not")
arg = parser.parse_args()
modes = ['train', 'dev', 'test']
task = ''
if arg.add_qstn: task = '-qstn'
for mode in modes:
    make_data(mode, task, arg.add_qstn)
