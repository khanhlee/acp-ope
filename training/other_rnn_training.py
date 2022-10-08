# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:55:51 2022

@author: khanhlee
"""

from __future__ import print_function, division
import os

import pandas as pd
import numpy as np

from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D,Reshape, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
from keras.models import load_model

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

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Bidirectional, TimeDistributed
from keras.layers.recurrent import SimpleRNN,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import Model
from keras.callbacks import EarlyStopping
import os
import tarfile
import re

pd.set_option('display.max_columns',None)

np.set_printoptions(threshold=np.inf)

"""Importing Data"""

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

"""Process the y"""

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

ACP_y_train_[0:2]

"""Transfer seq to protein_seq_dict for operation

Change protein_seq_dict to x_train
"""

x_train = {}
protein_index = 1
for line in mainTrain["Seq"]:
  x_train[protein_index] = line
  protein_index = protein_index + 1

print(len(x_train))
print([i for i in x_train.values()][:5])

#The longest length of train protein is 50
maxlen_train = max(len(x) for x in x_train.values())
maxlen_train

x_test = {}
protein_index = 1
for line in mainTest["Seq"]:
  x_test[protein_index] = line
  protein_index = protein_index + 1

print(len(x_test))
print([i for i in x_test.values()][:5])

#The longest length of test protein is 50
maxlen_test = max(len(x) for x in x_test.values())
maxlen_train

maxlen = max(maxlen_train,maxlen_train)
maxlen

#Convert amino acids to vectors
def OE(seq_temp):
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
  OE_feature = OE(x_train[i])
  x_train_bpf.append(OE_feature)
  #print(protein_seq_dict[i])

x_test_bpf = []
for i in x_test:
  OE_feature = OE(x_test[i])
  x_test_bpf.append(bpf_OE)

"""The longest protein sequence has 50 amino acids, each with 21 dimensions Complement the short sequence"""

x_train_OE = pad_sequences(x_train_OE, padding='post', maxlen=maxlen)
x_test_OE = pad_sequences(x_test_OE, padding='post', maxlen=maxlen)

x_train_OE[0]

x_train_OE.size()

"""Splitting the data set"""

X_train, X_val, y_train, y_val = train_test_split(x_train_OE, ACP_y_train_, test_size=0.2, random_state=1)

np.save('X_train.npy',X_train)

numpy_array = np.load('X_train.npy')

len(numpy_array)

len(X_train)

len(X_val)





x_test_bpf[0:2]

"""Training Model"""

vocab_size=22
embedding_dim=100

vocab_size=1051
embedding_dim=100

#DBRNN
def DBRNN(maxlen = 50, max_features = vocab_size, embed_size = embedding_dim):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(16, return_sequences=True), merge_mode='concat'))
    model.add(SimpleRNN(8))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

modeld = DBRNN()
modeld.summary()
modeld.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

batch_size = 64
epochs = 4
hist_drnn = modeld.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=batch_size,
          epochs=epochs,
          #callbacks=[es],
          shuffle=True)
loss, accuracy = modeld.evaluate(X_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = modeld.evaluate(X_val, y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist_drnn)

"""validation"""

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
import keras

def build_model():
    model = Sequential()
    #model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen,weights=[embedding_matrix],trainable=False))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen,trainable=True))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    #model.add(MaxPooling1D(pool_size=6,strides=6))
    #model.add(Dense(2, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  #loss='categorical_crossentropy',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

#DBRNN
def DBRNN(maxlen = 50, max_features = vocab_size, embed_size = embedding_dim):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(16, return_sequences=True), merge_mode='concat'))
    model.add(SimpleRNN(8))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
              #loss='categorical_crossentropy',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

model1 = KerasClassifier(build_fn=DBRNN,epochs=10, batch_size=256)

seed = 1

kfold = KFold(n_splits=5,shuffle=True,random_state=seed)

result = cross_val_score(model1,x_train_OE,ACP_y_train,cv=kfold)

print("============")
print("mean:",result.mean())
print("std:",result.std())



"""Other RNN model

RNN
"""

vocab_size=22
embedding_dim=100

vocab_size=1051
embedding_dim=100

def RNN(maxlen = 50, max_features = vocab_size, embed_size = embedding_dim):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(SimpleRNN(16))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = RNN()

model.summary()\

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5)

batch_size = 64
epochs = 7
hist_rnn = model.fit(x_train_OE, ACP_y_train_,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          #callbacks=[es],
          shuffle=True)

loss, accuracy = model.evaluate(x_train_OE, ACP_y_train_, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test_OE, ACP_y_test_, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist_rnn)

"""BRRN"""

def BRNN(maxlen = 50, max_features = vocab_size, embed_size = embedding_dim):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(16, return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

model = BRNN()

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5)

batch_size = 64
epochs = 10
hist_brnn = model.fit(x_train_OE, ACP_y_train_,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          #callbacks=[es],
          shuffle=True)

loss, accuracy = model.evaluate(x_train_OE, ACP_y_train_, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test_OE, ACP_y_test_, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

accuracy

plot_history(hist_brnn)



"""DBRNN"""

def DBRNN(maxlen = 50, max_features = vocab_size, embed_size = embedding_dim):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(16, return_sequences=True), merge_mode='concat'))
    model.add(SimpleRNN(8))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = DBRNN()

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5)

batch_size = 64
epochs = 11
hist_drnn = model.fit(x_train_OE, ACP_y_train_,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          #callbacks=[es],
          shuffle=True)

loss, accuracy = model.evaluate(x_train_OE, ACP_y_train_, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test_OE, ACP_y_test_, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist_drnn)



"""GRU and BIGRU"""

def GRU_model(maxlen = 50, max_features = vocab_size, embed_size = embedding_dim):
    print('Build GRU model...')
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def BIGRU_model(maxlen = 50, max_features = vocab_size, embed_size = embedding_dim): #双向
    print('Build BIGRU model...')
    model = Sequential()
    model.add(Embedding (max_features ,embed_size,input_length =maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(32,return_sequences =True),merge_mode ='concat'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1,activation= 'sigmoid'))
    model.compile(loss= 'binary_crossentropy',optimizer= 'adam',metrics= ['accuracy'])
    return  model

model = BIGRU_model()

model = GRU_model()

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5)

batch_size = 64
epochs = 8
hist_drnn = model.fit(x_train_OE, ACP_y_train_,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          #callbacks=[es],
          shuffle=True)

loss, accuracy = model.evaluate(x_train_OE, ACP_y_train_, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test_OE, ACP_y_test_, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(hist_drnn)







