# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:55:51 2022

@author: khanhlee
"""

from __future__ import print_function, division
import os
import re
import sys
import random

import pandas as pd
import numpy as np

from keras import layers
from keras import initializers
from keras.models import Sequential,load_model,Model
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D,Reshape, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate, LSTM, Bidirectional

from numpy import array,argmax,linalg as la
from numpy.linalg import eig

import sklearn
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from lightgbm.sklearn import LGBMClassifier 
#from sklearn.externals import joblib

import matplotlib.pyplot as plt
from plot_keras_history import plot_history

pd.set_option('display.max_columns',None)
np.set_printoptions(threshold=np.inf)

"""Data Input"""

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
mainTest = fasta2csv(inFastaTest)

"""handle y"""

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
ACP_y_train_ = np.array([np.array(i) for i in ACP_y_train])
ACP_y_test_ = np.array([np.array(i) for i in ACP_y_test])

"""get x_train_oe"""

x_train = {}
protein_index = 1
for line in mainTrain["Seq"]:
  x_train[protein_index] = line
  protein_index = protein_index + 1
maxlen_train = max(len(x) for x in x_train.values())

x_test = {}
protein_index = 1
for line in mainTest["Seq"]:
  x_test[protein_index] = line
  protein_index = protein_index + 1
maxlen_test = max(len(x) for x in x_test.values())

maxlen = max(maxlen_train,maxlen_test)

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

x_train_oe = []
for i in x_train:
  oe_feature = OE(x_train[i])
  x_train_oe.append(oe_feature)
  #print(protein_seq_dict[i])

x_test_oe = []
for i in x_test:
  oe_feature = OE(x_test[i])
  x_test_oe.append(oe_feature)

x_train_ = np.array(pad_sequences(x_train_oe, padding='post', maxlen=maxlen))
x_test_ = np.array(pad_sequences(x_test_oe, padding='post', maxlen=maxlen))

"""handcraft

AAC
"""

handcraft_AAC_train = [[0] * 21 for _ in range(len(x_train_oe))]
for row in range(len(x_train_oe)):
  seq = x_train_oe[row]
  for i in seq:
    col = i[0]-1
    handcraft_AAC_train[row][col] += 1/len(seq)
hc_AAC_train = np.array(handcraft_AAC_train)

handcraft_AAC_test = [[0] * 21 for _ in range(len(x_test_oe))]
for row in range(len(x_test_oe)):
  seq = x_test_oe[row]
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

handcraft_DPC_train = [[0] * len(comb) for _ in range(len(x_train_oe))]
for row in range(len(x_train_oe)):
  seq = x_train_oe[row]
  for i in range(len(seq)-1):
    a = sorted([seq[i][0],seq[i+1][0]])
    index = comb_index[tuple(a)]
    handcraft_DPC_train[row][index] += 1/(len(seq)-1)
hc_DPC_train = np.array(handcraft_DPC_train)

handcraft_DPC_test = [[0] * len(comb) for _ in range(len(x_test_oe))]
for row in range(len(x_test_oe)):
  seq = x_test_oe[row]
  for i in range(len(seq)-1):
    a = sorted([seq[i][0],seq[i+1][0]])
    index = comb_index[tuple(a)]
    handcraft_DPC_test[row][index] += 1/(len(seq)-1)
hc_DPC_test = np.array(handcraft_DPC_test)

"""CKS"""

def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" does not exist.')
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

"""kmer"""

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
def prepare_feature_kmer(infile):
  protein_seq_dict = {}
  protein_index = 1
  with open(infile, 'r') as fp:
    for line in fp:
      if line[0] != '>':
        seq = line[:-1]
        protein_seq_dict[protein_index] = seq
        protein_index = protein_index + 1
  kmer = []
  groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
  group_dict = TransDict_from_list(groups)
  protein_tris = get_3_protein_trids()
  # get protein feature
  # pdb.set_trace()
  for i in protein_seq_dict:  # and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
    protein_seq = translate_sequence(protein_seq_dict[i], group_dict)
    # print('oe:',shape(oe_feature))
    # pdb.set_trace()
    # RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
    protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
    kmer.append(protein_tri_fea)
    protein_index = protein_index + 1
    # chem_fea.append(chem_tmp_fea)
  return np.array(kmer)

kmer_train = prepare_feature_kmer(inFastaTrain)
kmer_test = prepare_feature_kmer(inFastaTest)

hc_train = np.c_[hc_AAC_train,hc_DPC_train,hc_CKS_train,kmer_train]
hc_train.shape

hc_test = np.c_[hc_AAC_test,hc_DPC_test,hc_CKS_test,kmer_test]
hc_test.shape

"""Train_test_split"""

X_train,X_val,y_train,y_val,HC_train,HC_val = train_test_split(x_train_,ACP_y_train_,hc_train,test_size=0.2,random_state=1)

HC_train.shape

"""Bilstm + new features"""

vocab_size=22
embedding_dim=100

main_input = Input((maxlen,),dtype='int32',name='main_input')
x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
lstm_out = Bidirectional(LSTM(64))(x)
aux_len = HC_train.shape[1]
aux_input = Input((aux_len,),name='aux_input')
x = layers.concatenate([lstm_out,aux_input])
x = Dropout(0.5)(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=[main_input,aux_input],outputs=main_output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#train the model
hist = model.fit(x={'main_input':X_train,'aux_input':HC_train}, y= y_train,
                    #class_weight = class_weights_d,
                    epochs=20,
                    verbose=True,
                    validation_data=([X_val,HC_val],y_val),
                    batch_size=64).history

loss, accuracy = model.evaluate(x={'main_input':X_train,'aux_input':HC_train}, y = y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x={'main_input':X_val,'aux_input':HC_val}, y = y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist)

"""BiLSTM(lstm)"""

main_input = Input((maxlen,),dtype='int32',name='main_input')
x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
lstm_out = LSTM(64)(x)
#lstm_out = Bidirectional(LSTM(64))(x)
x = Dropout(0.5)(lstm_out)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
#x = Dense(64,activation='relu')(x)
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=main_input,outputs=main_output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#train the model
hist = model.fit(x=X_train, y= y_train,
                    #class_weight = class_weights_d,
                    epochs=40,
                    verbose=True,
                    validation_data=(X_val,y_val),
                    batch_size=64).history

loss, accuracy = model.evaluate(x=X_train, y = y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x=X_val, y = y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist)

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
        # W_regularizer: 
        # b_regularizer: 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        # W_constraint: 
        # b_constraint: 
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

"""Attention + new features"""

main_input = Input(shape=(maxlen,), dtype='int32',name='main_input')
x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
x = Bidirectional(LSTM(100, return_sequences=True))(x)
x = Attention(maxlen)(x)
aux_len = HC_train.shape[1]
aux_input = Input((aux_len,),name='aux_input')
x = layers.concatenate([x,aux_input])
x = Dropout(0.5)(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
#x = Dense(64,activation='relu')(x)
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=[main_input,aux_input], outputs=main_output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

hist = model.fit(x={'main_input':X_train,'aux_input':HC_train}, y= y_train,
                    #class_weight = class_weights_d,
                    epochs=30,
                    verbose=True,
                    validation_data=([X_val,HC_val],y_val),
                    batch_size=64).history

loss, accuracy = model.evaluate(x={'main_input':X_train,'aux_input':HC_train}, y = y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x={'main_input':X_val,'aux_input':HC_val}, y = y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist)

"""Attention"""

vocab_size=22
embedding_dim=100
main_input = Input(shape=(maxlen,), dtype='int32',name='main_input')
x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
x = Bidirectional(LSTM(100, return_sequences=True))(x)
x = Attention(maxlen)(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=main_input, outputs=main_output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

hist = model.fit(x= X_train, y= y_train,
                    #class_weight = class_weights_d,
                    epochs=50,
                    verbose=True,
                    validation_data=(X_val,y_val),
                    batch_size=64).history

loss, accuracy = model.evaluate(x=X_train, y = y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x=X_val, y = y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist)

"""Feature Selection"""

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

"""RF selector"""

SF_ALL_K_train=RF_SelectFeatures(hc_train,ACP_y_train_,150)
SF_ALL_K_test=RF_SelectFeatures(hc_test,ACP_y_test_,150)

X_train,X_val,y_train,y_val,HS_train,HS_val = train_test_split(x_train_,ACP_y_train_,SF_ALL_K_train,test_size=0.2,random_state=1)

main_input = Input((maxlen,),dtype='int32',name='main_input')
x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
lstm_out = Bidirectional(LSTM(64))(x)
aux_len = HS_train.shape[1]
aux_input = Input((aux_len,),name='aux_input')
x = layers.concatenate([lstm_out,aux_input])
x = Dropout(0.5)(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=[main_input,aux_input],outputs=main_output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#train the model
hist = model.fit(x={'main_input':X_train,'aux_input':HS_train}, y= y_train,
                    #class_weight = class_weights_d,
                    epochs=30,
                    verbose=True,
                    validation_data=([X_val,HS_val],y_val),
                    batch_size=64).history

loss, accuracy = model.evaluate(x={'main_input':X_train,'aux_input':HS_train}, y = y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x={'main_input':X_val,'aux_input':HS_val}, y = y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist)

"""LGBM selector"""

SF_ALL_K_train=LGBM_SelectFeatures(hc_train,ACP_y_train_,150)
SF_ALL_K_test=LGBM_SelectFeatures(hc_test,ACP_y_test_,150)

X_train,X_val,y_train,y_val,HS_train,HS_val = train_test_split(x_train_,ACP_y_train_,SF_ALL_K_train,test_size=0.2,random_state=1)

main_input = Input((maxlen,),dtype='int32',name='main_input')
x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
lstm_out = Bidirectional(LSTM(64))(x)
aux_len = HS_train.shape[1]
aux_input = Input((aux_len,),name='aux_input')
x = layers.concatenate([lstm_out,aux_input])
x = Dropout(0.5)(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=[main_input,aux_input],outputs=main_output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#train the model
hist = model.fit(x={'main_input':X_train,'aux_input':HS_train}, y= y_train,
                    #class_weight = class_weights_d,
                    epochs=30,
                    verbose=True,
                    validation_data=([X_val,HS_val],y_val),
                    batch_size=64).history

loss, accuracy = model.evaluate(x={'main_input':X_train,'aux_input':HS_train}, y = y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x={'main_input':X_val,'aux_input':HS_val}, y = y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist)

"""pca"""

def pca(X,k):
  X = X - X.mean(axis = 0)
  X_cov = np.cov(X.T, ddof = 0)
  eigenvalues, eigenvectors = eig(X_cov)
  klarge_index = eigenvalues.argsort()[-k:][::-1]
  k_eigenvectors = eigenvectors[klarge_index]
  return np.dot(X,k_eigenvectors.T)

hc_train_pca = pca(hc_train,148)
hc_test_pca = pca(hc_test,148)

X_train,X_val,y_train,y_val,HS_train,HS_val = train_test_split(x_train_,ACP_y_train_,hc_train_pca,test_size=0.2,random_state=1)

main_input = Input((maxlen,),dtype='int32',name='main_input')
x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
lstm_out = Bidirectional(LSTM(64))(x)
aux_len = HS_train.shape[1]
aux_input = Input((aux_len,),name='aux_input')
x = layers.concatenate([lstm_out,aux_input])
x = Dropout(0.5)(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=[main_input,aux_input],outputs=main_output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#train the model
hist = model.fit(x={'main_input':X_train,'aux_input':HS_train}, y= y_train,
                    #class_weight = class_weights_d,
                    epochs=30,
                    verbose=True,
                    validation_data=([X_val,HS_val],y_val),
                    batch_size=64).history

loss, accuracy = model.evaluate(x={'main_input':X_train,'aux_input':HS_train}, y = y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x={'main_input':X_val,'aux_input':HS_val}, y = y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist)

"""data augmentation"""

def oversamp(X_result,oe, p):
    add_num = int(len(X_result)*p)
#    print(add_num)
#if(1):
    X_add_all = []
    oe_add_all = []
    for i in range(add_num):
        idx_ram = random.randint(0,X_result.shape[0]-1)
        oe_sel = oe[idx_ram,::]
        oe_add_all.append(oe_sel)
        X_sel = X_result[idx_ram,:]
        value = np.random.uniform(0, 1, (1, fea_num))        
#        value = np.random.normal(0,1,(1,fea_num)) 
#        value = np.random.poisson(6, size=(1,fea_num))
#        value = np.random.exponential(10, size=(1,fea_num))
        add_value = value*delta*X_sel
#        add_value[0,0] = 0 # ORFLen not be added
        X_add = X_sel + add_value
        X_add = np.squeeze(X_add)
        X_add_all.append(X_add)
    X_add_all = np.array(X_add_all)
    oe_add_all = np.array(oe_add_all)   
#    label_add = np.ones((add_num,),dtype = int)
    return X_add_all, oe_add_all

X_train,X_val,y_train,y_val,HC_train,HC_val = train_test_split(x_train_,ACP_y_train_,hc_train,test_size=0.2,random_state=1)

augtimes = 1
fea_num = HC_train.shape[1]
delta = 0.02
idx_pos = (y_train == 1)
idx_neg = (y_train == 0)
X_pos = HC_train[idx_pos,:]
X_neg = HC_train[idx_neg,:]
oe_pos = X_train[idx_pos,::]
oe_neg = X_train[idx_neg,::]
X_pos_add,oe_pos_add = oversamp(X_pos,oe_pos, augtimes)
X_neg_add,oe_neg_add = oversamp(X_neg,oe_neg, augtimes)
hc_train_aug = np.concatenate((HC_train,X_pos_add,X_neg_add))
x_train_oe_aug = np.concatenate((X_train,oe_pos_add,oe_neg_add))
ACP_y_train_aug = np.concatenate((y_train,y_train))

main_input = Input((maxlen,),dtype='int32',name='main_input')
x = Embedding(vocab_size, embedding_dim, input_length=maxlen,trainable=True)(main_input)
lstm_out = Bidirectional(LSTM(64))(x)
aux_len = hc_train_aug.shape[1]
aux_input = Input((aux_len,),name='aux_input')
x = layers.concatenate([lstm_out,aux_input])
x = Dropout(0.5)(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
model = Model(inputs=[main_input,aux_input],outputs=main_output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#train the model
hist = model.fit(x={'main_input':x_train_oe_aug,'aux_input':hc_train_aug}, y= ACP_y_train_aug,
                    #class_weight = class_weights_d,
                    epochs=50,
                    verbose=True,
                    validation_data=([X_val,HC_val],y_val),
                    batch_size=64).history

loss, accuracy = model.evaluate(x={'main_input':x_train_oe_aug,'aux_input':hc_train_aug}, y = ACP_y_train_aug, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x={'main_input':X_val,'aux_input':HC_val}, y = y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist)

import joblib

joblib.dump(model,filename='bilstm_attention_main7632.joblib')