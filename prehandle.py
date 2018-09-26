#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
# encoding='utf-8'
import pickle as cPickle
import numpy as np
import util

def main():
    good_text = []
    bad_text = []
    with open('normal_web.sql', mode='r', encoding='UTF-8') as fp:
        for line in fp:
            raw = util.filter(line)
            good_text.append(raw)
    with open('evil_web.sql', mode='r', encoding='UTF-8') as fp:
        for line in fp:
            raw = util.filter(line)
            bad_text.append(raw)
    vocab = util.build_vocab(good_text, bad_text)

    DATA_X = []
    DATA_Y = []

    print("len good text", len(good_text))
    print("len bad text", len(bad_text))
    

    bad_cnt = 0
    for line in bad_text:
        rst = util.encode(vocab, line)
        DATA_X.append(rst)
        DATA_Y.append([1,0])
        bad_cnt += 1

    for line in good_text:
        rst = util.encode(vocab, line)
        if bad_cnt > 0:
            DATA_X.append(rst)
            DATA_Y.append([0,1])
            bad_cnt -= 1

    cPickle.dump( [np.array(DATA_X), np.array(DATA_Y)] , open('./data.bin','wb') )
    cPickle.dump( vocab , open('./vocab.bin','wb') )

if __name__ == '__main__':
    main()
