# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:55:51 2022

@author: khanhlee
"""

import xgboost as xgb
import lightgbm as lgb  
from sklearn.model_selection  import train_test_split
from sklearn.metrics import *
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from numpy import array,argmax,linalg as la
from keras.preprocessing.sequence import pad_sequences
import os
import re

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

ACP_y_train

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
    #k = 6
    for i in range(len(seq)):
        if seq[i] =='A':
            tem_vec = 1
        elif seq[i]=='C':
            tem_vec = 2
        elif seq[i]=='D':
            tem_vec = 3
        elif seq[i]=='E' or seq[i]=='U':
            tem_vec = 4
        elif seq[i]=='F':
            tem_vec = 5
        elif seq[i]=='G':
            tem_vec = 6
        elif seq[i]=='H':
            tem_vec = 7
        elif seq[i]=='I':
            tem_vec = 8
        elif seq[i]=='K':
            tem_vec = 9
        elif seq[i]=='L':
            tem_vec = 10
        elif seq[i]=='M' or seq[i]=='O':
            tem_vec = 11
        elif seq[i]=='N':
            tem_vec = 12
        elif seq[i]=='P':
            tem_vec = 13
        elif seq[i]=='Q':
            tem_vec = 14
        elif seq[i]=='R':
            tem_vec = 15
        elif seq[i]=='S':
            tem_vec = 16
        elif seq[i]=='T':
            tem_vec = 17
        elif seq[i]=='V':
            tem_vec = 18
        elif seq[i]=='W':
            tem_vec = 19
        elif seq[i]=='X' or seq[i]=='B' or seq[i]=='Z':
            tem_vec = 20    
        elif seq[i]=='Y':
            tem_vec = 21
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

x_test_.shape

handcraft_AAC_train = [[0] * 21 for _ in range(len(x_train_oe))]
for row in range(len(x_train_oe)):
  seq = x_train_oe[row]
  for i in seq:
    col = i-1
    handcraft_AAC_train[row][col] += 1/len(seq)
hc_AAC_train = np.array(handcraft_AAC_train)

handcraft_AAC_test = [[0] * 21 for _ in range(len(x_test_oe))]
for row in range(len(x_test_oe)):
  seq = x_test_oe[row]
  for i in seq:
    col = i-1
    handcraft_AAC_test[row][col] += 1/len(seq)
hc_AAC_test = np.array(handcraft_AAC_test)

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
    a = sorted([seq[i],seq[i+1]])
    index = comb_index[tuple(a)]
    handcraft_DPC_train[row][index] += 1/(len(seq)-1)
hc_DPC_train = np.array(handcraft_DPC_train)

handcraft_DPC_test = [[0] * len(comb) for _ in range(len(x_test_oe))]
for row in range(len(x_test_oe)):
  seq = x_test_oe[row]
  for i in range(len(seq)-1):
    a = sorted([seq[i],seq[i+1]])
    index = comb_index[tuple(a)]
    handcraft_DPC_test[row][index] += 1/(len(seq)-1)
hc_DPC_test = np.array(handcraft_DPC_test)

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

X_train = np.c_[hc_train,x_train_]
X_test = np.c_[hc_test,x_test_]

AAC_TRAIN,AAC_VAL,DPC_TRAIN,DPC_VAL,CKS_TRAIN,CKS_VAL,KMER_TRAIN,KMER_VAL,HC_TRAIN,HC_VAL,OE_TRAIN,OE_VAL,X_TRAIN,X_VAL,Y_TRAIN,Y_VAL = train_test_split(hc_AAC_train,hc_DPC_train,hc_CKS_train,kmer_train,hc_train,x_train_,X_train,ACP_y_train_,test_size=0.2,random_state=4)

#XGBboost+AAC
xgb_model = xgb.XGBClassifier()
xgb_model.fit(AAC_TRAIN,Y_TRAIN)
xgb_predict = xgb_model.predict(AAC_VAL)

print("xgb val ACC：" ,str(xgb_model.score(AAC_VAL,Y_VAL)))
print("xgb train ACC：" ,str(xgb_model.score(AAC_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict)))
print("precision_score:",str(precision_score(Y_VAL,xgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,xgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,xgb_predict,average = 'weighted')))

#XGBboost+DPC
xgb_model = xgb.XGBClassifier()
xgb_model.fit(DPC_TRAIN,Y_TRAIN)
xgb_predict = xgb_model.predict(DPC_VAL)

print("xgb val ACC：" ,str(xgb_model.score(DPC_VAL,Y_VAL)))
print("xgb train ACC：" ,str(xgb_model.score(DPC_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict)))
print("precision_score:",str(precision_score(Y_VAL,xgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,xgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,xgb_predict,average = 'weighted')))

#XGBboost+CKS
xgb_model = xgb.XGBClassifier()
xgb_model.fit(CKS_TRAIN,Y_TRAIN)
xgb_predict = xgb_model.predict(CKS_VAL)

print("xgb val ACC：" ,str(xgb_model.score(CKS_VAL,Y_VAL)))
print("xgb train ACC：" ,str(xgb_model.score(CKS_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,xgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,xgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,xgb_predict,average = 'weighted')))

#XGBboost+kmer
xgb_model = xgb.XGBClassifier()
xgb_model.fit(KMER_TRAIN,Y_TRAIN)
xgb_predict = xgb_model.predict(KMER_VAL)

print("xgb VAL ACC：" ,str(xgb_model.score(KMER_VAL,Y_VAL)))
print("xgb train ACC：" ,str(xgb_model.score(KMER_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,xgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,xgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,xgb_predict,average = 'weighted')))

#XGBboost+hc
xgb_model = xgb.XGBClassifier()
xgb_model.fit(HC_TRAIN,Y_TRAIN)
xgb_predict = xgb_model.predict(HC_VAL)

print("xgb VAL ACC：" ,str(xgb_model.score(HC_VAL,Y_VAL)))
print("xgb train ACC：" ,str(xgb_model.score(HC_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,xgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,xgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,xgb_predict,average = 'weighted')))

#XGBboost+oe
xgb_model = xgb.XGBClassifier()
xgb_model.fit(OE_TRAIN,Y_TRAIN)
xgb_predict = xgb_model.predict(OE_VAL)

print("xgb VAL ACC：" ,str(xgb_model.score(OE_VAL,Y_VAL)))
print("xgb train ACC：" ,str(xgb_model.score(OE_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,xgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,xgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,xgb_predict,average = 'weighted')))

#XGBboost+all
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_TRAIN,Y_TRAIN)
xgb_predict = xgb_model.predict(X_VAL)

print("xgb ACC：" ,str(xgb_model.score(X_VAL,Y_VAL)))
print("xgb train ACC：" ,str(xgb_model.score(X_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,xgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,xgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,xgb_predict,average = 'weighted')))

#lightgbm+AAC
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(AAC_TRAIN,Y_TRAIN)
lgb_predict = lgb_model.predict(AAC_VAL)

print("lgb VAL ACC：" ,str(lgb_model.score(AAC_VAL,Y_VAL)))
print("LGB train ACC：" ,str(lgb_model.score(AAC_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,lgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,lgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,lgb_predict,average = 'weighted')))

#lightgbm+DPC
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(DPC_TRAIN,Y_TRAIN)
lgb_predict = lgb_model.predict(DPC_VAL)

print("lgb ACC：" ,str(lgb_model.score(DPC_VAL,Y_VAL)))
print("LGB train ACC：" ,str(lgb_model.score(DPC_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,lgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,lgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,lgb_predict,average = 'weighted')))

#lightgbm+CKS
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(CKS_TRAIN,Y_TRAIN)
lgb_predict = lgb_model.predict(CKS_VAL)

print("lgb ACC：" ,str(lgb_model.score(CKS_VAL,Y_VAL)))
print("LGB train ACC：" ,str(lgb_model.score(CKS_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,lgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,lgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,lgb_predict,average = 'weighted')))

#lightgbm+kmer
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(KMER_TRAIN,Y_TRAIN)
lgb_predict = lgb_model.predict(KMER_VAL)

print("lgb ACC：" ,str(lgb_model.score(KMER_VAL,Y_VAL)))
print("LGB train ACC：" ,str(lgb_model.score(KMER_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,lgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,lgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,lgb_predict,average = 'weighted')))

#lightgbm+oe
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(OE_TRAIN,Y_TRAIN)
lgb_predict = lgb_model.predict(OE_VAL)

print("lgb ACC：" ,str(lgb_model.score(OE_VAL,Y_VAL)))
print("LGB train ACC：" ,str(lgb_model.score(OE_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,lgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,lgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,lgb_predict,average = 'weighted')))

#lightgbm+hc
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(HC_TRAIN,Y_TRAIN)
lgb_predict = lgb_model.predict(HC_VAL)

print("lgb ACC：" ,str(lgb_model.score(HC_VAL,Y_VAL)))
print("LGB train ACC：" ,str(lgb_model.score(HC_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,lgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,lgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,lgb_predict,average = 'weighted')))

#lightgbm+all
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_TRAIN,Y_TRAIN)
lgb_predict = lgb_model.predict(X_VAL)

print("lgb ACC：" ,str(lgb_model.score(X_VAL,Y_VAL)))
print("LGBtrain ACC：" ,str(lgb_model.score(X_TRAIN,Y_TRAIN)))
#print("roc_auc_score:",str(roc_auc_score(y_te,xgb_predict))) 不支持多分类
print("precision_score:",str(precision_score(Y_VAL,lgb_predict,average = 'weighted')))
print("recall_score:" , str(recall_score(Y_VAL,lgb_predict,average = 'weighted')))
print("f1_score:",str(f1_score(Y_VAL,lgb_predict,average = 'weighted')))

#import joblib

#joblib.dump(lgb_model,filename='lgbm_main7865.joblib')
#joblib.dump(xgb_model,filename='xgboost_main7661.joblib')

#model1 = joblib.load(filename='lgbm_main7865.joblib')

#model1.predict(X_test)

from sklearn.datasets import *
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

classifier = svm.SVC(kernel="rbf")
classifier.fit(OE_TRAIN, Y_TRAIN)
pre_train=classifier.predict(OE_TRAIN)
pre_test=classifier.predict(OE_VAL)
print(accuracy_score(Y_TRAIN, pre_train))
print(accuracy_score(Y_VAL, pre_test))

classifier = svm.SVC(kernel="poly")
classifier.fit(OE_TRAIN, Y_TRAIN)
pre_train=classifier.predict(OE_TRAIN)
pre_test=classifier.predict(OE_VAL)
print(accuracy_score(Y_TRAIN, pre_train))
print(accuracy_score(Y_VAL, pre_test))

classifier = svm.SVC(kernel="sigmoid")
classifier.fit(OE_TRAIN, Y_TRAIN)
pre_train=classifier.predict(OE_TRAIN)
pre_test=classifier.predict(OE_VAL)
print(accuracy_score(Y_TRAIN, pre_train))
print(accuracy_score(Y_VAL, pre_test))

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,GridSearchCV

forest = RandomForestClassifier()
forest.fit(X_TRAIN, Y_TRAIN)
result = forest.score(X_VAL,Y_VAL)
result