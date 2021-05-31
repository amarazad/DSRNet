#!/usr/bin/env python
# coding: utf-8

def get_key(my_dict, val, doc_id):
    for key, value in my_dict.items(): 
         if val in value and key.split('__')[0] == doc_id: 
            return key 
  
    print(my_dict, val, doc_id)
    return "key doesn't exist"


tmp = []
with open('resources/bad_words.txt', 'r') as f:
    tmp = f.readlines()
bad_word = {word.strip('\n') : True for word in tmp}

#method to identify foul words
def find_bad(word):
    try:
        return bad_word[word]
    except:
        return False


tmp = []
with open('resources/ubuntu_commands.txt', 'r') as f:
    tmp = f.readlines()
cmd = {word.strip('\n') : True for word in tmp}

#method to identify ubuntu commands
def find_command(word):
    try:
        return cmd[word]
    except:
        return False

#method to load pretrained question detection model
import pickle
def get_Qdetector():
    f_name = open("questionDetector.pickle", "rb")
    classifier_q = pickle.load(f_name)
    f_name.close()
    return classifier_q

from nltk.tokenize import word_tokenize 
def dialogue_act_features(post):
    features = {}
    for word in word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


classifier_q = get_Qdetector()

#method to identify utterances which are questions
def get_question(sentence):

    Question_Words = ['what', 'where', 'when','how','why','did','do','does','have','has','am','is','are','can','could','may','would','will','should'
        "didn't","doesn't","haven't","isn't","aren't","can't","couldn't","wouldn't","won't","shouldn't",'?']
    Question_Words_Set = set(Question_Words)
    w_5w1h = ['what','who', 'why','when','where','how']
    w_vw = ['can', 'could', 'are', 'is','am','did','has','would']
    imp = ['please','kindly']
    Q_words_Set = set( w_5w1h+ w_vw + imp )  

    if len(sentence) <= 0 or sentence == ' ' or sentence == '': return False # if blank after Q_filter
    
    import re
    #remove elements of utterance that may contain ?
    url = r'[a-zA-z]+://[^\s]*'
    sentence = re.sub(url, "URL", sentence)
    url = r'www\.[^\s]*'
    sentence = re.sub(url, "URL", sentence)
    filepath = r'\w*/\w+/?[^\s]*'
    sentence = re.sub(filepath, "FILEPATH", sentence)
    email = r'\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*'
    sentence = re.sub(email, 'EMAIL', sentence)
    sentence = re.sub(r"[`+~@#$%\^&*\|\\/<>]", "", sentence)
    sentence = re.sub(r"\.\.\.", "", sentence)
    command = r'\s-[a-zA-Z]*'
    sentence = re.sub(command, ' COMMAND', sentence)  

    if "?" in sentence:
        return True
    tokens = word_tokenize(sentence.lower())

    #checking if the first word of the sentence is a qstn word
    if Q_words_Set.intersection([tokens[0]]): 
        return True
    
    if Question_Words_Set.intersection(tokens) == False:
        return False

    predicted = classifier_q.classify(dialogue_act_features(sentence))
    
    if predicted == 'Question':
        return True
        
    return False 


#stats
def get_stats(conv, min_turn, mode='train'):
    import numpy as np
    print('==========Statistics for ', mode, ' :================')
    print('No. of conversations = ', len([turn for turn in list(conv.values()) if len(turn) >= min_turn]))
    print('Max turn length in conversations = ', max([len(turn) for turn in list(conv.values()) if len(turn) >= min_turn]))
    print('Average turn length in conversations = ', np.mean([len(turn) for turn in list(conv.values()) if len(turn) >= min_turn]))
    print('Min turn length in conversations = ', min_turn)
    
#create text files
import string
trans = str.maketrans(string.punctuation, ' '*len(string.punctuation) )

from langdetect import detect
from fuzzywuzzy import fuzz

#main method for preprocessing and creating data files
def get_data(conversation, corpus, docs_path, hist_len=3,  add_q=False, add_t=False, sw=True):
    min_turn = hist_len
    
    instances = []
    names = {}
    docs = os.listdir(docs_path)
    for doc in tqdm(docs):
        
        if doc.split('.')[-1] == 'txt' and doc.split('.')[-2] == 'ascii':
            doc_path = os.path.join(docs_path, doc)

            #fetch conversations with turn > hist_len -- sliding window :
            for k, v in conversation.items():
                if doc.split('.')[0] == k.split('__')[0] and len(v) >= hist_len:
                    with open(doc_path) as fin:
                        lines = fin.readlines()
                    prev = ''
                    prev_q = ''
                    qstn = ['N']*hist_len

                    for idx in v:
                        
                        try:
                            name = lines[int(idx)-1].split('<')[1].split('>')[0].strip()
                        except:
                            name = ''

                        
                        if name and name.lower() != 'linux' and name.lower() != 'ubuntu' : names[name.lower()] = True

                        def find_names(w):
                            try:
                                return names[w]
                            except:
                                try:
                                    return names[w.replace(',', '')]
                                except:
                                    pass
                                if ':' in w : 
                                    return True
                                if '<' and '>' in w:
                                    return True
                            return False
                        try:
                            #removing usernames and bad words
                            line = ' '.join([word for word in lines[int(idx)-1].split('>')[1].strip('\n').split() if not find_bad(word)]).strip()
                        except:
                            if lines[int(idx)-1][0] == '=':
                                continue
                            
                            else:
                                #removing bad words
                                line = ' '.join([word for word in lines[int(idx)-1].split(']')[1].strip('\n').split() if not find_bad(word)])

                        #removing usernames (from front and end or in commands)
                        line = ' '.join([word for num, word in enumerate(line.split()) if not find_names(word.lower().strip()) ])
                        if ' | ' in line : line = line.split(' | ')[0].strip()
                        
                        
                        #removing lines like ....
                        if line.translate(trans).strip() == '':
                            continue
                        
                        #avoiding message repetition:
                        if fuzz.ratio(line, prev) >= 90:
                            continue 
                        
                        #removing one word utterances which are not questions or ubuntu commands -attempt to remove chit chat kind of utterances
                        if len(line.split()) == 1 and not get_question(line) and not find_command(line.translate(trans).strip()):
                            continue
                        
                        #removing non-english utterances
                        try:
                            if line and detect(line) != 'en':
                                continue
                        except:
                            pass
                        
                        if hist_len > 1 and len(prev.split('[eos]')) <= hist_len - 1:
                            if prev == '': 
                                prev = line
                                if get_question(line) : qstn[0] = 'Y'
                            else:
                                if get_question(line) : qstn[len(prev.split('[eos]'))] = 'Y'
                                prev = prev + ' [eos] ' + line
                            continue
                        
                        elif hist_len == 1 and prev == '':
                            prev = line
                            if get_question(line) : qstn[0] = 'Y'
                            continue
                        else:
                            if add_q: 
                                instances.append(prev + ' [eos]  [eoc] ' + '-'.join(qstn) + ' [eoq] [sep] ' + line + ' [eos]')
                            else:
                                instances.append(prev + ' [eos]  [sep] ' + line + ' [eos]')
                            #sliding window
                            if sw:
                                if len(prev.split('[eos]')) == hist_len:
                                    prev = ' [eos] '.join(prev.split('[eos]')[1:]) + ' [eos] ' + line
                                    if add_q : 
                                        if get_question(line): 
                                            qstn = qstn[1:] + ['Y']
                                        else:
                                            qstn = qstn[1:] + ['N']
                                else:
                                    prev = prev.strip('\n') + ' [eos] ' + line
                                    if add_q : 
                                        if get_question(line): qstn[len(prev.split('[eos]'))] = 'Y'
                            else:
                                prev = prev.strip('\n') + ' [eos] ' + line
                                if add_q : 
                                    if get_question(line): qstn[-1] = 'Y'

                        

    if add_q : string = corpus + '-qstnYN'
    if not add_q : string = corpus
    with open('kummerfeld/ctxt-'+string+'.txt', 'w' ) as f:
        print('No. of datapoints = ', len(instances))
        for line in instances:
            f.write('%s\n' %line.replace('[eos]  [eos]', '[eos]').strip())



import os
from tqdm import tqdm

#method for conversation disentanglement
def format(hist_len, add_q, add_t, sliding):
    
    english_corpus = ['train', 'dev', 'test']
    corpus_path = 'kummerfeld/data/'

    for en_corpus in english_corpus:
        docs_path = os.path.join(corpus_path, en_corpus)
        docs = os.listdir(docs_path)

        conversation = {}
        conv_id = 0
        
        for doc in tqdm(docs):
            visited = []

            if doc.split('.')[-1] == 'txt' and doc.split('.')[-2] == 'annotation':
                doc_path = os.path.join(docs_path, doc)
                conversation[doc.split('.')[0]+'__'+str(conv_id)] = []

                with open(doc_path) as fin:
                    for line in fin.readlines():
                        line = line.strip()

                        prev_talk = line.split()[0]
                        next_talk = line.split()[1]

                        if prev_talk not in conversation[doc.split('.')[0]+'__'+str(conv_id)]:
                            if prev_talk  not in visited:
                                conv_id += 1
                                if prev_talk == next_talk: #standalone, single utterance conversation
                                    conversation[doc.split('.')[0]+'__'+str(conv_id)] = [next_talk]
                                else:
                                    conversation[doc.split('.')[0]+'__'+str(conv_id)] = [prev_talk, next_talk]

                            else:
                                #handling one utterance reply to multiple prev utterances
                                if conversation[get_key(conversation,str(prev_talk), doc.split('.')[0])][-1] == next_talk: 
                                    continue
                                conversation[get_key(conversation,str(prev_talk), doc.split('.')[0])].append(next_talk)                           


                        else:
                            #handling one utterance reply to multiple prev utterances
                            if conversation[doc.split('.')[0]+'__'+str(conv_id)][-1] == next_talk: 
                                continue
                            else:
                                conversation[doc.split('.')[0]+'__'+str(conv_id)].append(next_talk)

                        visited.append(prev_talk)
                        visited.append(next_talk)
                        
        #get statistics
        get_stats(conversation, hist_len, en_corpus)
        get_data(conversation, en_corpus, docs_path, hist_len, add_q, add_t, sliding)
                        


import argparse
    
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--turn_len", default=3, type=int, help="Mention no. of turns in dialogue history")
parser.add_argument("--add_qstn", default=False, type=bool,help="Whether to add question or not")
parser.add_argument("--sliding_win", default=True, type=bool,help="Whether to add sliding window or not")
arg = parser.parse_args()
format(arg.turn_len, arg.add_qstn, False, arg.sliding_win)