#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
# encoding='utf-8'

import os,random
import numpy as np
import pickle as cPickle
import util
import keras

if __name__ == '__main__':
    # Load dataset
    DATA_X, DATA_Y = cPickle.load(open("./data.bin",'rb'),  encoding='latin1')
    print(DATA_X.shape)
    print(DATA_Y.shape)

    randomize = np.arange(len(DATA_X))
    np.random.shuffle(randomize)
    DATA_X_TRAIN = DATA_X[randomize[0:int(len(DATA_X)/2)]]
    DATA_Y_TRAIN = DATA_Y[randomize[0:int(len(DATA_X)/2)]]
    DATA_X_TEST = DATA_X[randomize[int(len(DATA_X)/2):len(DATA_X)]]
    DATA_Y_TEST = DATA_Y[randomize[int(len(DATA_X)/2):len(DATA_X)]]

    # Load Neural Network Model
    model = util.init_model()

    # Training
    history = model.fit(DATA_X_TRAIN, DATA_Y_TRAIN,
        batch_size = util.batch_size,
        epochs     = util.epoch,
        verbose    = 2,
        validation_data=(DATA_X_TEST, DATA_Y_TEST),
        callbacks = [
            keras.callbacks.ModelCheckpoint(util.filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1),
        ])

    # Dump Model
    with open('train.log','w') as f:
        f.write(str(history.history))
