#!/usr/bin/env python
# coding: utf-8

import nltk
import json
import argparse
import csv, re
import pickle
from os import listdir
from os.path import isfile, join

from nltk.tokenize import word_tokenize 


import nltk.classify
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def train_Qdtector(posts):
    
    featuresets = [(dialogue_act_features(post.text), 'Question') if post.get('class') == 'whQuestion' or post.get('class') == 'ynQuestion' else (dialogue_act_features(post.text), 'O') for post in posts]
    types = set([post.get('class') for post in posts])
    print(types)
    for i, post in enumerate(posts):
        if i < 5:
            print(i, post.text, post.get('class'))
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
    y_pred = []
    y_true = []
    for tup in test_set:
        y_pred.append(classifier.classify(tup[0]))
        y_true.append(tup[1])
    print(classification_report(y_true, y_pred, digits=4))
    
    return classifier

def save_question_detector():
    nltk.download('nps_chat')
    posts = nltk.corpus.nps_chat.xml_posts()
    classifier = train_Qdtector(posts)
    filename = 'questionDetector.pickle'
    pickle.dump(classifier, open(filename, 'wb'))
    print(filename, "saved ..")


def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


if __name__ == "__main__":
    save_question_detector()
