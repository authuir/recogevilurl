#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
# encoding='utf-8'
import pickle as cPickle
import numpy as np
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D

max_line_len  = 1024
train_mode    = True
dr            = 0.5
epoch         = 100                         # number of epochs to train on
batch_size    = 10                         # training batch size
filepath      = 'test.wts.h5'

def init_model():
    # Construct Model
    model = Sequential()
    model.add( Reshape([1]+[max_line_len], input_shape=[max_line_len]) )
    model.add(LSTM(units=128))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def filter(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str+i
    return content_str

def build_vocab(good_text, bad_text):
    code = int(0)
    vocab = {}
    vocab[u'夨'] = code
    code += 1
    for lines in good_text:
        for wrd in lines:
            if not wrd in vocab:
                vocab[wrd] = code
                code += 1
    
    for lines in bad_text:
        for wrd in lines:
            if not wrd in vocab:
                vocab[wrd] = code
                code += 1

    return vocab

def encode(vocab, string):
    x = []
    words = string
    for i in range(0, max_line_len):
        if (i < len(words)):
            if words[i] in vocab:
                x.append(vocab[words[i]])
            else:
                x.append(vocab[u'夨'])
                print("cannot recognize:",words[i])
        else:
            x.append(vocab[u'夨'])
    return x
