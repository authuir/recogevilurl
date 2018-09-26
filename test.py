#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
# encoding='utf-8'
import pickle as cPickle
import numpy as np
import requests
import keras
import util
import sys

model = util.init_model()
model.load_weights(util.filepath)

def handle(url):
    r = requests.get(url)
    text = util.filter(str(r.content,'utf-8'))

    vocab = cPickle.load(open("./vocab.bin",'rb'),  encoding='latin1')
    X_test = util.encode(vocab, text)

    Y_test = model.predict(np.array(X_test).reshape([1,util.max_line_len]))
    if Y_test[0][0] > Y_test[0][1]:
        print("Evil URL ,", url)
    else:
        print("Normal URL,", url)
    #print(text)
    print("概率： ", max(Y_test[0][0] , Y_test[0][1]))

def main():
    url = "http://baidu.com/"
    if len(sys.argv) == 2:
        url = sys.argv[1]
    handle(url)


if __name__ == '__main__':
    #main()
    testlist = ['http://www.danshiqi.com','http://www.tytz9.com','http://www.dyshicheng.com','http://www.jiansheng88.com','http://www.dayuefund.net','http://www.younaidp.com','http://92.lui66sy.xyz/pw/','http://8460.shuadan99.com/','https://0fffxx.com/fanhao/newest#header']
    for it in testlist:
        handle(it)