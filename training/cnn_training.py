# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:55:51 2022

@author: khanhlee
"""

from __future__ import print_function, division
import os

import pandas as pd
import numpy as np
import random

from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D,Conv2D, GlobalMaxPooling1D,GlobalMaxPooling2D,Reshape, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate,LSTM, Bidirectional
from keras.models import load_model,Model

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#from sklearn.externals import joblib

import matplotlib.pyplot as plt
from plot_keras_history import plot_history

import re

import tensorflow as tf
#from tensorflow.keras.regularizers import l2
from numpy import linalg as la
from lightgbm.sklearn import LGBMClassifier 
from sklearn.ensemble import RandomForestClassifier

from keras.layers import LSTM

import torch
from torch import nn
import math

# pd.set_option('display.max_columns',None)

# np.set_printoptions(threshold=np.inf)

"""Import Data"""

def parse_stream(f, comment=b'#'):
    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                yield name, b''.join(sequence)
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        yield name, b''.join(sequence)

def fasta2csv(inFasta):
    FastaRead=pd.read_csv(inFasta,header=None)
    print(FastaRead.shape)
    print(FastaRead.head())
    seqNum=int(FastaRead.shape[0]/2)
    csvFile=open("testFasta.csv","w")
    csvFile.write("PID,Seq\n")
    
    #print("Lines:",FastaRead.shape)
    #print("Seq Num:",seqNum)
    for i in range(seqNum):
      csvFile.write(str(FastaRead.iloc[2*i,0])+","+str(FastaRead.iloc[2*i+1,0])+"\n")
            
         
    csvFile.close()
    TrainSeqLabel=pd.read_csv("testFasta.csv",header=0)
    path="testFasta.csv"
    if os.path.exists(path):
     
        os.remove(path)  
     
    return TrainSeqLabel

inFastaTrain="/data/ACP20mainTrain.fasta"
inFastaTest="/data/ACP20mainTest.fasta"

mainTrain = fasta2csv(inFastaTrain)

mainTrain.head()

mainTest = fasta2csv(inFastaTest)

mainTest.head()

"""process y"""

i=0
mainTrain["Tags"]=mainTrain["Seq"]
for pid in mainTrain["PID"]:
  mainTrain["Tags"][i]=pid[len(pid)-1]
  if mainTrain["Tags"][i]=="1":
    mainTrain["Tags"][i]=1
  else:
    mainTrain["Tags"][i]=0
  i=i+1

i=0
mainTest["Tags"]=mainTest["Seq"]
for pid in mainTest["PID"]:
  mainTest["Tags"][i]=pid[len(pid)-1]
  if mainTest["Tags"][i]=="1":
    mainTest["Tags"][i]=1
  else:
    mainTest["Tags"][i]=0
  i=i+1

ACP_y_train = mainTrain["Tags"].values
ACP_y_test = mainTest["Tags"].values

ACP_y_train_ = [np.array(i) for i in ACP_y_train]
ACP_y_test_ = [np.array(i) for i in ACP_y_test]

ACP_y_train_ = np.array(ACP_y_train_)
ACP_y_test_ = np.array(ACP_y_test_)

"""protein_seq_dict=>x_train"""

x_train = {}
protein_index = 1
for line in mainTrain["Seq"]:
  x_train[protein_index] = line
  protein_index = protein_index + 1

x_test = {}
protein_index = 1
for line in mainTest["Seq"]:
  x_test[protein_index] = line
  protein_index = protein_index + 1

#The longest length of train protein is 50
maxlen_train = max(len(x) for x in x_train.values())
print(maxlen_train)

#The longest length of test protein is 50
maxlen_test = max(len(x) for x in x_test.values())
print(maxlen_test)

maxlen = max(maxlen_train,maxlen_train)
maxlen

"""Add features"""

#Convert amino acids to vectors
def OE(seq_temp):
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
    fea = []
    tem_vec =[]
    #k = 6
    for i in range(len(seq)):
        if seq[i] =='A':
            tem_vec = [1]
        elif seq[i]=='C':
            tem_vec = [2]
        elif seq[i]=='D':
            tem_vec = [3]
        elif seq[i]=='E' or seq[i]=='U':
            tem_vec = [4]
        elif seq[i]=='F':
            tem_vec = [5]
        elif seq[i]=='G':
            tem_vec = [6]
        elif seq[i]=='H':
            tem_vec = [7]
        elif seq[i]=='I':
            tem_vec = [8]
        elif seq[i]=='K':
            tem_vec = [9]
        elif seq[i]=='L':
            tem_vec = [10]
        elif seq[i]=='M' or seq[i]=='O':
            tem_vec = [11]
        elif seq[i]=='N':
            tem_vec = [12]
        elif seq[i]=='P':
            tem_vec = [13]
        elif seq[i]=='Q':
            tem_vec = [14]
        elif seq[i]=='R':
            tem_vec = [15]
        elif seq[i]=='S':
            tem_vec = [16]
        elif seq[i]=='T':
            tem_vec = [17]
        elif seq[i]=='V':
            tem_vec = [18]
        elif seq[i]=='W':
            tem_vec = [19]
        elif seq[i]=='X' or seq[i]=='B' or seq[i]=='Z':
            tem_vec = [20]    
        elif seq[i]=='Y':
            tem_vec = [21]
        #fea = fea + tem_vec +[i]
        fea.append(tem_vec)
    return fea

x_train_OE = []
for i in x_train:
  OE_feature = OE(x_train[i])
  x_train_OE.append(OE_feature)
  #print(protein_seq_dict[i])

x_test_OE = []
for i in x_test:
  OE_feature = OE(x_test[i])
  x_test_OE.append(OE_feature)

"""AAC"""

handcraft_AAC_train = [[0] * 21 for _ in range(len(x_train_OE))]
for row in range(len(x_train_OE)):
  seq = x_train_OE[row]
  for i in seq:
    col = i[0]-1
    handcraft_AAC_train[row][col] += 1/len(seq)
hc_AAC_train = np.array(handcraft_AAC_train)

handcraft_AAC_test = [[0] * 21 for _ in range(len(x_test_OE))]
for row in range(len(x_test_OE)):
  seq = x_test_OE[row]
  for i in seq:
    col = i[0]-1
    handcraft_AAC_test[row][col] += 1/len(seq)
hc_AAC_test = np.array(handcraft_AAC_test)

"""DPC"""

comb = []
for i in range(1,22):
  for j in range(i,22):
    comb.append([i,j])
comb_index = {}
for i in range(len(comb)):
  comb_index[tuple(comb[i])] = i
handcraft_DPC_train = [[0] * len(comb) for _ in range(len(x_train_OE))]
for row in range(len(x_train_OE)):
  seq = x_train_OE[row]
  for i in range(len(seq)-1):
    a = sorted([seq[i][0],seq[i+1][0]])
    index = comb_index[tuple(a)]
    handcraft_DPC_train[row][index] += 1/(len(seq)-1)
hc_DPC_train = np.array(handcraft_DPC_train)

handcraft_DPC_test = [[0] * len(comb) for _ in range(len(x_test_OE))]
for row in range(len(x_test_OE)):
  seq = x_test_OE[row]
  for i in range(len(seq)-1):
    a = sorted([seq[i][0],seq[i+1][0]])
    index = comb_index[tuple(a)]
    handcraft_DPC_test[row][index] += 1/(len(seq)-1)
hc_DPC_test = np.array(handcraft_DPC_test)

"""CKSAAGP"""

import re
import sys
def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" dOEs not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) == None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])

    return myFasta

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair


def CKSAAGP(fastas, gap = 5, **kw):

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1+'.'+key2)

    encodings = []
    header = ['#']
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p+'.gap'+str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] = gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)

        encodings.append(code)

    return encodings

handcraft_CKSAAGP_train = CKSAAGP(readFasta(inFastaTrain))
handcraft_CKS_train = []
for i in range(1,len(handcraft_CKSAAGP_train)):
  handcraft_CKS_train.append(handcraft_CKSAAGP_train[i][1:])
hc_CKS_train = np.array(handcraft_CKS_train)

handcraft_CKSAAGP_test = CKSAAGP(readFasta(inFastaTest))
handcraft_CKS_test = []
for i in range(1,len(handcraft_CKSAAGP_test)):
  handcraft_CKS_test.append(handcraft_CKSAAGP_test[i][1:])
hc_CKS_test = np.array(handcraft_CKS_test)

"""K-mer"""

def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)  # Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index])  # K:V map, use group's first letter as represent.
        index = index + 1
    return result

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com

def translate_sequence(seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    import string
    from_list = []
    to_list = []
    for k, v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    # TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq

def get_4_nucleotide_composition(tris, seq, pythoncount=True):
    seq_len = len(seq)
    tri_feature = [0] * len(tris)
    k = len(tris[0])
    note_feature = [[0 for cols in range(len(seq) - k + 1)] for rows in range(len(tris))]
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num) / seq_len)
    else:
        # tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                # tmp_fea[ind] = tmp_fea[ind] + 1
                note_feature[ind][x] = note_feature[ind][x] + 1
        # tri_feature = [float(val)/seq_len for val in tmp_fea]    #tri_feature type:list len:256
        u, s, v = la.svd(note_feature)
        for i in range(len(s)):
            tri_feature = tri_feature + u[i] * s[i] / seq_len
        # print tri_feature
        # pdb.set_trace()

    return tri_feature

def prepare_feature_kmer(protein_seq_dict):
    label = []
    interaction_pair = {}
    RNA_seq_dict = {}

    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()

    kmer = []

    for i in protein_seq_dict:
        protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)

        kmer.append(protein_tri_fea)

    return np.array(kmer)

kmer_train=prepare_feature_kmer(x_train)
kmer_test=prepare_feature_kmer(x_test)

"""4 Features:745"""

hc_train = np.c_[hc_AAC_train,hc_DPC_train,hc_CKS_train,kmer_train]
hc_train.shape

hc_test = np.c_[hc_AAC_test,hc_DPC_test,hc_CKS_test,kmer_test]
hc_test.shape

X=hc_train

"""four feature augment"""

fea_num=745

def oversamp_pos(X_result, p):
  delta = 0.05
  add_num = int(len(X_result)*p)
  X_add_all = []

  for i in range(add_num):
    idx_ram = random.randint(0,X_result.shape[0]-1)
    X_sel = X_result[idx_ram,:]
    value = np.random.uniform(0, 1, (1, fea_num)) 
#    value = np.concatenate((value1, value2),axis = 1)
#    value = np.random.normal(0,1) 
#    value = np.random.poisson(6, size=(1,402)) 
#    value = np.random.exponential(10, size=(1,402)) 
    add_value = value*delta*X_sel

    X_add = X_sel + add_value
    X_add = np.squeeze(X_add)
    X_add_all.append(X_add)

  X_add_all = np.array(X_add_all)   
  return X_add_all#,label_add

def oversamp_neg(X_result, p):
  delta = 0.05
  add_num = int(len(X_result)*p)
  X_add_all = []

  for i in range(add_num):
    idx_ram = random.randint(0,X_result.shape[0]-1)
    X_sel = X_result[idx_ram,:]
    value = np.random.uniform(0, 1, (1, fea_num))  
#    value = np.concatenate((value1, value2),axis = 1)
#    value = np.random.normal(0,1) 
#    value = np.random.poisson(6, size=(1,402)) 
#    value = np.random.exponential(10, size=(1,402)) 
    add_value = value*delta*X_sel
    X_add = X_sel + add_value
    X_add = np.squeeze(X_add)
    X_add_all.append(X_add)

  X_add_all = np.array(X_add_all)    
  return X_add_all#,label_add

def ACP_DL():
    np.random.seed(0)
    random.seed(0)
    augtimes = 1

    train = X
    train_label = ACP_y_train_

    # augment the train data
    idx_pos = (train_label == 1)
    idx_neg = (train_label == 0)
    X_pos = train[idx_pos,:]
    X_neg = train[idx_neg,:]
    X_pos_add = oversamp_pos(X_pos, augtimes)
    X_neg_add = oversamp_neg(X_neg, augtimes)

    X_origin = np.concatenate((X_pos,X_neg))
    X_new = np.concatenate((X_pos_add,X_neg_add))
    X_aug = np.concatenate((X_origin,X_new))

    label_aug = np.hstack([train_label]*2)

    return X_aug,label_aug

X_aug,label_aug=ACP_DL()

X_aug.shape

"""add position info in OE"""

#Convert amino acids to vectors
def OE_position(seq_temp):
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
    fea = []
    tem_vec =[]
    #k = 6
    for i in range(len(seq)):
        if seq[i] =='A':
            tem_vec = [0*50+i]
        elif seq[i]=='C':
            tem_vec = [1*50+i]
        elif seq[i]=='D':
            tem_vec = [2*50+i]
        elif seq[i]=='E' or seq[i]=='U':
            tem_vec = [3*50+i]
        elif seq[i]=='F':
            tem_vec = [4*50+i]
        elif seq[i]=='G':
            tem_vec = [5*50+i]
        elif seq[i]=='H':
            tem_vec = [6*50+i]
        elif seq[i]=='I':
            tem_vec = [7*50+i]
        elif seq[i]=='K':
            tem_vec = [8*50+i]
        elif seq[i]=='L':
            tem_vec = [9*50+i]
        elif seq[i]=='M' or seq[i]=='O':
            tem_vec = [10*50+i]
        elif seq[i]=='N':
            tem_vec = [11*50+i]
        elif seq[i]=='P':
            tem_vec = [12*50+i]
        elif seq[i]=='Q':
            tem_vec = [13*50+i]
        elif seq[i]=='R':
            tem_vec = [14*50+i]
        elif seq[i]=='S':
            tem_vec = [15*50+i]
        elif seq[i]=='T':
            tem_vec = [16*50+i]
        elif seq[i]=='V':
            tem_vec = [17*50+i]
        elif seq[i]=='W':
            tem_vec = [18*50+i]
        elif seq[i]=='X' or seq[i]=='B' or seq[i]=='Z':
            tem_vec = [19*50+i]    
        elif seq[i]=='Y':
            tem_vec = [20*50+i]
        #tem_vec = tem_vec +[i]
        fea.append(tem_vec)
        #fea.append(tem_vec)
    return fea

x_train_OE = []
for i in x_train:
  OE_feature = OE_position(x_train[i])
  x_train_OE.append(OE_feature)

x_test_OE = []
for i in x_test:
  OE_feature = OE_position(x_test[i])
  x_test_OE.append(OE_feature)

"""The longest protein sequence has 50 amino acids, and each amino acid has 21 dimensions. Complete the short sequences."""

x_train_OE = pad_sequences(x_train_OE, padding='post', maxlen=maxlen)
x_test_OE = pad_sequences(x_test_OE, padding='post', maxlen=maxlen)

"""copy OE"""

x_train_OE_aug = np.vstack([x_train_OE]*2)

len(x_train_OE_aug)

"""Select Features"""

def RF_SelectFeatures(X,y,i):#RadomForest Feature Selection Method

    model = RandomForestClassifier(n_estimators=888,min_samples_leaf=10,random_state=2020,n_jobs=8)
    model.fit(X, y)
    importantFeatures = model.feature_importances_
    K = importantFeatures.argsort()[::-1][:i]
    RF_ALL_K=X[:,K]

    return  RF_ALL_K

def LGBM_SelectFeatures(X,y,i):#Light Gradient Boosting Machine Feature Selection

    model = LGBMClassifier(num_leaves=32,n_estimators=888,max_depth=12,learning_rate=0.16,min_child_samples=50,random_state=2020,n_jobs=8)
    model.fit(X, y)
    importantFeatures = model.feature_importances_
    K = importantFeatures.argsort()[::-1][:i]
    LGB_ALL_K=X[:,K]

    return  LGB_ALL_K

"""Attention"""

'''
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.layers import Layer'''

from keras import backend as K, initializers, regularizers, constraints
from keras.layers import Layer

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
 
        if self.bias:
                    self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
 
        self.built = True
 
    def compute_mask(self, input, input_mask=None):
        return None
 
    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
 
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
 
        if self.bias:
            eij += self.b
 
        eij = K.tanh(eij)
 
        a = K.exp(eij)
 
        '''
        keras.backend.cast(x, dtype): 
        '''
        if mask is not None:
            a *= K.cast(mask, K.floatx())
 
        '''
        keras.backend.epsilon(): 
        '''
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon()   , K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
 
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

"""Train the model"""

X_train, X_val, y_train, y_val,add_train,add_val = train_test_split(x_train_OE, ACP_y_train_,hc_train, test_size=0.2, random_state=1)

vocab_size=1051
embedding_dim=100

def CNN_SF(m,n,tag):
  for i in range(m,n):
    print("Select %s features"%(i))
    if tag == 'RF':
      SF_ALL_K_train=RF_SelectFeatures(add_train,y_train,i)
      SF_ALL_K_test=RF_SelectFeatures(add_val,y_val,i)
    elif tag == 'LGBM':
      SF_ALL_K_train=LGBM_SelectFeatures(add_train,y_train,i)
      SF_ALL_K_test=LGBM_SelectFeatures(add_val,y_val,i)

    main_input = Input((maxlen,),dtype='int32',name='main_input')
    x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
    x = Conv1D(256, 3, activation='relu')(x)
    x_out = GlobalMaxPooling1D()(x)
    #x = MaxPooling1D(pool_size=2,strides=1)(x)
    #x = Bidirectional(LSTM(64,return_sequences=True))(x) #add Bi-LSTM
    #x_out = Attention(47)(x) #add Attention
    aux_input = Input((i,),name='aux_input')
    x = layers.concatenate([x_out,aux_input]) 
    x = Dense(64,activation='relu')(x)
    #x = Dense(64,activation='tanh')(x)
    x = Dropout(0.5)(x)
    main_output = Dense(1,activation='sigmoid',name='main_output')(x)
    #main_output = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation='linear')(x)
    model = Model(inputs=[main_input,aux_input],outputs=main_output)
    #model = Model(inputs=main_input,outputs=main_output)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    #model.compile('adam', 'hinge', metrics=['accuracy'])
    #model.summary()

    #train the model

    hist = model.fit(x={'main_input':X_train,'aux_input':SF_ALL_K_train}, y= y_train,
                    #class_weight = class_weights_d,
                    epochs=7,
                    verbose=False,
                    validation_data=([X_val,SF_ALL_K_test],y_val),
                    batch_size=64).history

    loss, accuracy = model.evaluate(x={'main_input':X_train,'aux_input':SF_ALL_K_train}, y = y_train, verbose=True)
    #print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x={'main_input':X_val,'aux_input':SF_ALL_K_test}, y = y_val, verbose=True)
    #print("Testing Accuracy:  {:.4f}".format(accuracy))


CNN_SF(140,150,'RF')