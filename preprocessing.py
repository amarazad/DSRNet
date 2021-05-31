#!/usr/bin/env python
# coding: utf-8

import sys
import os
import re
import json
from tqdm import tqdm
import copy
import nltk
from nltk.tokenize import word_tokenize

punct = [',', ':', ';']

from nltk.corpus import words

with open('resources/ubuntu_commands.txt', 'r') as f:
    commands = [line.split()[0] for line in f.readlines() if line.split()[0] != '.' and line.split()[0] not in words.words()]    

print(commands)


with open('resources/1000_common_words.txt', 'r') as f:
    common_words = [line.strip() for line in f.readlines()]
common_words.append('linux')
common_words.append('ubuntu')
common_words = list(set(common_words))[:500]
print(common_words)


more_stop_words = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english') + common_words + more_stop_words)

print(len(stop_words))
print(stop_words)


english_corpus = ['train','dev','test']
corpus_path = 'kummerfeld/data'
store_path ='kummerfeld/Corpus_processed/disentangle_data'
combine_path = 'kummerfeld/Corpus_processed/combined_disentangle_data'
session_path = 'kummerfeld/Corpus_processed/context_window_disentangle_15'

if not os.path.isdir(os.path.join(store_path, 'train')):
    os.makedirs(os.path.join(store_path, 'train'))
if not os.path.isdir(os.path.join(store_path, 'dev')):
    os.makedirs(os.path.join(store_path, 'dev'))
if not os.path.isdir(os.path.join(store_path, 'test')):
    os.makedirs(os.path.join(store_path, 'test'))

if not os.path.isdir(os.path.join(combine_path, 'train')):
    os.makedirs(os.path.join(combine_path, 'train'))
if not os.path.isdir(os.path.join(combine_path, 'dev')):
    os.makedirs(os.path.join(combine_path, 'dev'))
if not os.path.isdir(os.path.join(combine_path, 'test')):
    os.makedirs(os.path.join(combine_path, 'test'))

if not os.path.isdir(os.path.join(session_path, 'train')):
    os.makedirs(os.path.join(session_path, 'train'))
if not os.path.isdir(os.path.join(session_path, 'dev')):
    os.makedirs(os.path.join(session_path, 'dev'))
if not os.path.isdir(os.path.join(session_path, 'test')):
    os.makedirs(os.path.join(session_path, 'test'))


# filter sentences, number them, numbering starts from 1
def filter(uttr_min_word, en_corpus):
    threads =  []
    regexp = re.compile(r'^[\x20-\x7E]+$')
    thread_index = 1
    docs_path = os.path.join(corpus_path, en_corpus)
    docs = os.listdir(docs_path)
    for doc in docs:
        if doc.split('.')[-1] == 'txt' and doc.split('.')[-2] == 'ascii':
            doc_path = os.path.join(docs_path, doc)
            with open(doc_path) as fin:
                thread = []
                index  = 1
                for line in fin.readlines():
                    line = line.strip()
                    #if line not starting with '[' then they are system messages which are ignored by the following if statement 
                    if line.startswith('[') and regexp.search(line) is not None:
                        
                        text = line.split(' ')
                        if not(len(text) > 3 and len(text[2:]) > uttr_min_word  and text[1] != '*' and re.match(r'[^\s]', text[1][1:-1])):
                            index += 1
                            continue    
                        Id = "doc_" + str(thread_index) + '_' + str(index)
                        sender = text[1]
                        time = text[0]
                        thread.append([Id, time, sender, ' '.join(text[2:])])
                        index += 1
                    else:  
                        Id = "doc_" + str(thread_index) + '_' + str(index)
                        sender = '[system_message]'
                        time = '90:00'
                        thread.append([Id, time, sender, line])
                        index += 1
            
            threads.append(thread)
            thread_index += 1
    with open(os.path.join(store_path, en_corpus, 'all_data.txt'), 'w+') as fw:
        json.dump(threads, fw)

    return threads


def find_addressee(uttr_min_word, en_corpus):
    if not os.path.exists(os.path.join(store_path, en_corpus, 'all_data.txt')):
        threads = filter(uttr_min_word, en_corpus)
    else:
        with open(os.path.join(store_path, en_corpus, 'all_data.txt'), 'r') as f:
            threads = json.load(f)
    docs = []
    for thread in threads:
        doc = []
        sender_list = set()
        for line in thread:
            Id = line[0]
            time = line[1]
            sender = line[2][1:-1]
            sender_list.add(sender.lower())
            text = line[3].split(' ')
            if len(text[0]) > 1 and text[0][-1] in punct and text[0][:-1].lower() in sender_list:
                addressee = text[0][:-1]
            else:
                addressee = '-'
            doc.append([Id, time, sender, addressee, ' '.join(text) if addressee == '-' else ' '.join(text[1:])])
        docs.append(doc)
    os.remove(os.path.join(store_path, en_corpus, 'all_data.txt'))
    with open(os.path.join(store_path, en_corpus, 'data_with_addr.txt'), 'w+') as fw:
        json.dump(docs, fw)
    return docs


import string

def sentence_process(sentence):
    sentence = sentence.lower()
    emoji = r'[:;=]-?[()$PDp]'
    sentence = re.sub(emoji, "EMOJI", sentence)
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
    
    #tokenize
    sentence_list = word_tokenize(sentence)
    sentence_list = [word for word in sentence_list if word.translate(str.maketrans('', '', string.punctuation)) not in stop_words and not word.strip().isdigit() and len(word) > 3] #if you remove the last condition you will get all the punctuations back
    sentence_list = ['COMMAND' if word in commands else word for word in sentence_list]
    sentence = " ".join(sentence_list)
    desired_pos = ['NN', 'NNP', 'NNS', 'NNPS', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    pos_tags = [tup[1] for tup in nltk.pos_tag(sentence_list)]
    flag = 0
    for pos in pos_tags:
        if pos in desired_pos:
            flag = 1
            break
    if flag == 1:
        #combine with space
        sentence = " ".join(sentence_list)
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    else:
        sentence = ""
    
    return sentence


def filter_all_sentence(en_corpus, context_window_size=15, uttr_min_word=5):
    print('filter_all_sentence')
    if not os.path.exists(os.path.join(store_path, en_corpus, 'data_with_addr.txt')):
        docs = find_addressee(uttr_min_word, en_corpus)
    else:
        with open(os.path.join(store_path, en_corpus, 'data_with_addr.txt')) as f:
            print('else')
            docs = json.load(f)
    doc_index = 1
    for doc in tqdm(docs):
        
        if len(doc) < context_window_size:
            doc_index += 1
            continue
        with open(os.path.join(store_path, en_corpus, 'doc_' + str(doc_index)), 'w+') as fw:
            fw.write('Id' + '\t' + 'time' + '\t' + 'Sender_name' + '\t' + 'Addressee_name' + '\t' + 'Utterance' + '\n')
            for line in doc:
                line[4] = sentence_process(line[4])
                if len(line[4].split()) <= uttr_min_word  or not re.match(r'[^\s]', line[4]):
                    continue
                fw.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3]  + '\t' + line[4] + '\n')
        doc_index += 1
    
    return docs


#merge same user utterance
def combine_agent(en_corpus):
    print('combine_agent')
    docs = os.listdir(store_path+'/'+en_corpus)
    for doc in docs:
        if doc.split('_')[0] != 'doc':
            continue
        with open(os.path.join(store_path, en_corpus, doc)) as f:
            with open(os.path.join(combine_path, en_corpus, 'combined_' + doc), 'w+') as fw:
                thread = f.readlines()
                i = 0
                while(i < len(thread)):
                    if i == 0:
                        fw.write(thread[i].strip() + '\n')
                        i += 1
                        continue
                    j = i + 1
                    line = thread[i].strip().split('\t')
                    
                    if line[2] == 'system_message':
                        i += 1
                        continue
                    while(j < len(thread)):
                        combine_line = thread[j].strip().split('\t')
                        if (line[1] == combine_line[1] or line[3] != '-') and                            line[2] == combine_line[2] and line[3] == combine_line[3]:
                            line[-1] = line[-1] + ' ' + combine_line[-1]
                            j += 1
                        else:
                            break
                    i = j
                    fw.write('\t'.join(line) + '\n')


if __name__ == "__main__":
    context_window_size = 15
    # Threshold for preliminary screening
    uttr_min_word = 3
    #en_corpus = 'train'
    for en_corpus in english_corpus:
        filter_all_sentence(en_corpus, 15, uttr_min_word)
        combine_agent(en_corpus)

