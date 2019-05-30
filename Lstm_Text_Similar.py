#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@Date    : 2019-04-29
@Author  : Kaka
@File    : Lstm_Text_Similar.py
@Software: PyCharm
@Contact :
@Desc    : 文本相似度匹配：用于自动客服进行问题匹配

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import pandas as pd
import numpy as np
import networkx as nx
import jieba

from gensim.models import word2vec
import keras
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, concatenate, Embedding, Dropout, Dense, BatchNormalization
from keras import Model
from configparser import ConfigParser

jieba.add_word('花呗')
jieba.add_word('借呗')
jieba.add_word('蚂蚁借呗')
jieba.add_word('蚂蚁花呗')
jieba.add_word('支付宝')
jieba.add_word('余额宝')
jieba.add_word('是不是')
jieba.add_word('不是')
jieba.add_word('怎么还款')
jieba.add_word('怎么开通')
jieba.add_word('还能')
jieba.add_word('开不了')
jieba.add_word('开通不了')
jieba.add_word('要还')

class utils(object):
    @staticmethod
    def combine_csv(path1, path2):
        '''meger data'''
        dt1 = pd.read_csv(path1, header=None, index_col=0, sep='\t')
        dt2 = pd.read_csv(path2, header=None, index_col=0, sep='\t')
        dt = pd.concat([dt1, dt2], axis=0)
        dt.columns = ['q1', 'q2', 'y']
        return dt

    @staticmethod
    def ecode_text(path):
       pass

    @staticmethod
    def graph_text(gdata):
        pass

    @staticmethod
    def load_stop_synonym_words(stop_words_p, similar_words_p):
        ''''''
        stop_words = [line.strip() for line in open(stop_words_p, 'r', encoding='utf-8').readlines()]
        similar_words = [line.strip() for line in open(similar_words_p, 'r', encoding='utf-8').readlines()]
        return stop_words, similar_words

    @staticmethod
    def split_words(s, stop_words, similar_words):
        '''
        cut,rmstop,replace similar words
        '''
        word_split = ''
        sentence_list = jieba.cut(s, cut_all=False, HMM=True)
        for word in sentence_list:
            for similar_word in similar_words:
                if word in similar_word.split(' '):
                    word = similar_word.split(' ')[0].strip()
            if word not in stop_words:
                if word != '\t':
                    word_split += word.strip() + ' '
        return word_split.strip()

    class WordVector(object):
        def __init__(self, windows, embedding_dim, w2v_path, max_words_num):
            self.windows = windows
            self.embedding_dim = embedding_dim
            self.max_words_num = max_words_num
            self.w2v_path = w2v_path

        def fit_w2vector(self, data):
            print(data.head(5))
            data_all = data['q1'].append(data['q2'], ignore_index=True).to_frame(name='s')
            print(data_all.head(5))
            data_all['s'] = data_all['s'].apply(lambda x: str(x).split(' '))
            print(data_all['s'].head(5))
            for window in self.windows:
                model = word2vec.Word2Vec(data_all['s'].values, window=window, size=self.embedding_dim, )
                model.wv.save_word2vec_format(self.w2v_path + str(window), binary=False)

        def get_embedding_matrix(self, data, word_index, window=5):
            embeddings_index = {}
            if not os.path.exists(self.w2v_path + str(window)):
                self.fit_w2vector(data)
            with open(os.path.join(self.w2v_path + str(window)), 'r') as vf:
                for line in vf.readlines():
                    values = line.split(' ')
                    word = str(values[0])
                    embedding = np.asarray(values[1:], dtype='float')
                    embeddings_index[word] = embedding
            print('word embedding', len(embeddings_index))

            max_words_num = min(self.max_words_num, len(word_index))
            word_embedding_matrix = np.zeros((max_words_num + 1, self.embedding_dim))
            for word, i in word_index.items():
                if i > max_words_num:
                    continue
                embedding_vector = embeddings_index.get(str(word))
                if embedding_vector is not None:
                    word_embedding_matrix[i] = embedding_vector
            return word_embedding_matrix,max_words_num


class ProcessorTextMatch(object):
    '''processor'''

    def __init__(self, data_dir, train_file, valid_file, stop_words_p, similar_words_p, max_seq_len,max_words_num):
        self.data_dir = data_dir
        self.train_file = train_file
        self.valid_file = valid_file
        self.stop_words_p = stop_words_p
        self.similar_words_p = similar_words_p
        self.max_seq_len = max_seq_len
        self.max_words_num = max_words_num

    def process_data(self):
        data = utils.combine_csv(os.path.join(self.data_dir, self.train_file),
                                 os.path.join(self.data_dir, self.valid_file))
        stop_words, similar_words = utils.load_stop_synonym_words(os.path.join(self.data_dir, self.stop_words_p),
                                                                  os.path.join(self.data_dir, self.similar_words_p))
        data['q1'] = data['q1'].apply(lambda s: utils.split_words(s, stop_words, similar_words))
        data['q2'] = data['q2'].apply(lambda s: utils.split_words(s, stop_words, similar_words))

        ''''''
        tokenizer = Tokenizer(num_words=self.max_words_num)
        tokenizer.fit_on_texts(data['q1'].append(data['q2']))

        word_index = tokenizer.word_index
        self.max_words_num = min(self.max_words_num, len(word_index))

        '''texts_to_sequences'''
        x1 = tokenizer.texts_to_sequences(data['q1'])
        x2 = tokenizer.texts_to_sequences(data['q2'])

        '''padding'''
        x1 = pad_sequences(x1, maxlen=self.max_seq_len, padding='post', truncating='post')
        x2 = pad_sequences(x2, maxlen=self.max_seq_len, padding='post', truncating='post')

        return data, (x1, x2), data['y'].values, word_index


class TextMatchModel(object):
    def __init__(self, lstm_units, drop_rate, max_words_num, embedding_dim, max_seq_len, activation, epochs, batch_size,
                 bst_model_path):
        self.units = lstm_units
        self.drop_rate = drop_rate
        self.max_words_num = max_words_num
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.bst_model_path = bst_model_path

    def build_lstm(self, word_embedding_matrix):
        embedding_layer = Embedding(self.max_words_num + 1,
                                    self.embedding_dim,
                                    weights=[word_embedding_matrix],
                                    input_length=self.max_seq_len,
                                    trainable=False)

        input1 = Input(shape=(self.max_seq_len,))
        embed1 = embedding_layer(input1)
        x1 = LSTM(self.units, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(embed1)

        input2 = Input(shape=(self.max_seq_len,))
        embed2 = embedding_layer(input2)
        x2 = LSTM(self.units, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(embed2)

        merged = concatenate([x1, x2])
        merged = Dropout(self.drop_rate)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(64, activation=self.activation)(merged)
        merged = Dropout(self.drop_rate)(merged)
        merged = BatchNormalization()(merged)

        output = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[input1, input2], outputs=output)
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer='adam',
                      metrics=['acc', 'binary_crossentropy'])
        model.summary()
        plot_model(model, to_file='{}.png'.format("lstm_text_match"))
        return model

    def fit_model(self, X, y, model):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=False,verbose=1,)

        hist = model.fit([X[0], X[1]], y, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size,
                         shuffle=True,
                         callbacks=[early_stopping, model_checkpoint])

        bst_score = min(hist.history['loss'])
        bst_acc = max(hist.history['acc'])
        print(bst_acc, bst_score)


if __name__ == '__main__':
    cfp = ConfigParser()
    cfp.read('config.cfg')

    ptm = ProcessorTextMatch(data_dir=cfp['processor']['data_dir'],
                             train_file=cfp['processor']['train_file'],
                             valid_file=cfp['processor']['valid_file'],
                             stop_words_p=cfp['processor']['stop_words_p'],
                             similar_words_p=cfp['processor']['similar_words_p'],
                             max_seq_len=int(cfp['model']['max_seq_len']),
                             max_words_num=int(cfp['processor']['max_words_num']),
                             )
    utils_wv = utils.WordVector(windows=eval(cfp['word2vec']['windows']),
                                embedding_dim=int(cfp['word2vec']['embedding_dim']),
                                max_words_num=int(cfp['processor']['max_words_num']),
                                w2v_path=cfp['word2vec']['w2v_path'])


    data, X, y, word_index = ptm.process_data()
    embedding_matrix,max_words_num = utils_wv.get_embedding_matrix(data=data, word_index=word_index, window=5)

    tx_model = TextMatchModel(lstm_units=int(cfp['model']['units']),
                              drop_rate=float(cfp['model']['drop_rate']),
                              max_words_num=max_words_num,
                              embedding_dim=int(cfp['word2vec']['embedding_dim']),
                              max_seq_len=int(cfp['model']['max_seq_len']),
                              activation=cfp['model']['activation'],
                              epochs=int(cfp['model']['epochs']),
                              batch_size=int(cfp['model']['batch_size']),
                              bst_model_path=cfp['model']['bst_model_path']
                              )

    model = tx_model.build_lstm(embedding_matrix)
    tx_model.fit_model(X,y,model)