# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 01:22:25 2019

@author: kimomo
"""

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense, Dropout
import random
import os
import sys


def Data_prerpocessing_label1():
    # load data
    file_drug = pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM1_ESM.csv', header=None) #drug data
    file_drug = file_drug.rename({0:'drug'}, axis=1)
    file_target = pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM2_ESM.csv', header=None) #target data
    file_target = file_target.rename({0:'target'}, axis=1)
    file_drug_mat = pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM6_ESM.csv', header=None) #drug vector
    file_target_mat = pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM7_ESM.csv', header=None) #target vector
    file_interact_mat = pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM3_ESM.csv', header=None) #interact vector
    
    # check if column or row is all zero
    for i in file_interact_mat.columns:
        if sum(file_interact_mat[i])==0:
            file_interact_mat.drop(i, axis=1)
    for i in range(len(file_interact_mat)):
        if sum(file_interact_mat.iloc[i])==0:
            file_interact_mat.drop(i)
        
    data = []
    col ,row = np.where(file_interact_mat==1)
    for i in tqdm(range(len(row))):
        temp = []
        temp.append(file_drug.iloc[col[i]][0])
        temp.append(file_target.iloc[row[i]][0])
        temp.append(list(file_drug_mat.iloc[col[i]])+list(file_target_mat.iloc[row[i]]))
        temp.append(1)
        data.append(temp)
    df = pd.DataFrame(data, columns = ['drug' , 'target', 'data', 'label'])
    file = open('/Users/shanjuyeh/Desktop/Project/DLproject/new/output/data_label_1.pickle', 'wb')
    pickle.dump(df, file)
    file.close()

def Data_prerpocessing_label0(num):
    
    #load data
    DRUG_FEATURE=pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM6_ESM.csv', header=None).values.reshape(5877,193)
    TARGET_FEATURE=pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM7_ESM.csv', header=None).values.reshape(3348,1290)
    Target_name = pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM2_ESM.csv', header=None)
    Target_name = Target_name.rename({0:'target'}, axis=1)
    Drug_name = pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM1_ESM.csv', header=None)
    Drug_name = Drug_name.rename({0:'drug'}, axis=1)
    interaction_matrix=pd.read_csv('/Users/shanjuyeh/Desktop/Project/DLproject/12859_2016_1377_MOESM3_ESM.csv', header=None).values
    

    # get interaction equal 0
    binding_drug_site,binding_target_site=np.where(interaction_matrix==0)
    drugName=[]
    targetName=[]
    data=[]
    label=[]
    x = random.sample(range(len(binding_drug_site)),  num) # random sample from interaction 0
    
    for s in x:
        i=binding_drug_site[s]
        j=binding_target_site[s]

        drugName.append(Drug_name.values[i][0])
        targetName.append(Target_name.values[j][0])

        data.append(list(DRUG_FEATURE[i,:])+list(TARGET_FEATURE[j,:]))
        label.append(interaction_matrix[i,j])
    
    DB=pd.DataFrame({'drug':drugName,'target':targetName,'data':data,'label':label})
    
    filename = 'Label_zeros'+str(num)+'.pickle'
    
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(DB)
    n_bytes = sys.getsizeof(bytes_out)
    with open('/Users/shanjuyeh/Desktop/Project/DLproject/new/output/'+ filename, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filename):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filename)
        bytes_in = bytearray(0)
        with open(filename, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        Label_zeros = pickle.loads(bytes_in)
    except:
        return None
    return Label_zeros


def Oversampling(filename, num):
    file = open('/Users/shanjuyeh/Desktop/Project/DLproject/new/output/' + filename, 'rb')
    DB = pickle.load(file)
    
    lst = []
    for item in DB.data:
        lst.append(np.array(item))
    
    array_label_1 = np.stack(lst)
    clf = KMeans(n_clusters=num)
    clf.fit(array_label_1)
    labels = clf.labels_
    labels = list(labels)

    DB.label=labels
    Count=[labels.count(i) for i in range(num)]
    print('Count:',Count)
        
    max_count = max(Count)

    for i in range(num):
        DB=DB.append(DB[DB.label==i].sample(n=max_count-Count[i],replace=True,random_state=1))
    
    Count22=[list(DB.label).count(qq) for qq in range(num)]
    print('Count22:',Count22)
    DB.label = np.ones(len(DB))
    DB.index = np.arange(len(DB))
    
    #store data
    file = open('/Users/shanjuyeh/Desktop/Project/DLproject/new/output/LABEL_ONE_OVERSAMPLE_cluster100.pickle', 'wb')
    pickle.dump(DB, file)
    file.close()
    
    return len(DB)

def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0).reshape(1483,1)
    new =[]
    for i in range(len(meanVal)):
        tmp = dataMat[:,i]-meanVal[i]
        new.append(tmp)
    new = np.stack(new)
    return new.T,meanVal


def pca(dataMat,n):
    newData,meanVal = zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)    
    
    if os.path.isfile('C:\\Users\\kimomo\\Desktop\\DLproject\\PCA_matrix.pickle'):
        file = open('PCA_matrix.pickle', 'rb')
        n_eigVect = pickle.load(file)
        file.close()        
    else:
        eigVals,eigVects=np.linalg.eig(np.mat(covMat))
        eigValIndice=np.argsort(eigVals)            
        n_eigValIndice=eigValIndice[-1:-(n+1):-1]   
        n_eigVect=eigVects[:,n_eigValIndice]   
        
        file = open('PCA_matrix_GC.pickle', 'wb')
        pickle.dump(n_eigVect,file)
        file.close()
        
    lowDDataMat=newData*n_eigVect               
    print(n_eigVect.shape)
    
    # store transfer matrix
    file = open('n_eigVect.pickle', 'wb')
    pickle.dump(n_eigVect, file)
    file.close()
  
    return lowDDataMat    

def Data_concatenate_PCA(filename_1, filename_2):
    #load data
    file = open(filename_1, 'rb')
    data_label1 = pickle.load(file)
   
    data_label0 = try_to_load_as_pickled_object_or_None(filename_2)
    
    data_1 = []
    for item in data_label1.data:
        data_1.append(np.array(item))
    data_1 = np.stack(data_1)
    
    data_0 = []
    for item in data_label0.data:
        data_0.append(np.array(item))
    data_0 = np.stack(data_0)
    
    #data stack
    data_all = np.vstack([data_0, data_1])    
    lowDDataMat = pca(data_all, 1000)
    
    # data to dataframe
    DB=pd.DataFrame()
    DB['Drug']=list(data_label0.drug)+list(data_label1.drug)
    DB['Target']=list(data_label0.target)+list(data_label1.target)
    DB['Data']=[i for i in lowDDataMat]
    DB['Label']=list(data_label0.label)+list(data_label1.label)
    
    # store data
    filename = 'Data_preprocessed.pickle'
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(DB)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filename, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
            

def train_model(filename_3):

    data = try_to_load_as_pickled_object_or_None(filename_3)
    train = data.sample(frac=0.70,random_state=200)
    test = data.drop(train.index)
    
    X_train =[]
    for i in train.Data:
        X_train.append(i)
    X_train = np.stack(X_train)
    y_train = train.Label.values
    
    X_test =[]
    for i in test.Data:
        X_test.append(i)
    X_test = np.stack(X_test)
    y_test = test.Label.values
    
    # Store X_test y_test
    file = open('X_test.pickle', 'wb')
    pickle.dump(X_test, file)
    file.close()
    
    file = open('y_test.pickle', 'wb')
    pickle.dump(y_test, file)
    file.close()
    
    # Store X_train y_train
    file = open('X_train.pickle', 'wb')
    pickle.dump(X_train, file)
    file.close()
    
    file = open('y_train.pickle', 'wb')
    pickle.dump(y_train,file)
    file.close()
    
    DNN(X_train, y_train, X_test, y_test)
    
    return X_train, y_train, X_test, y_test
    
def DNN(X_train, y_train, X_test, y_test):
    model=Sequential()
    model.add(Dense(units=1024,input_dim=1000,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units=512,kernel_initializer='RandomNormal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256,kernel_initializer='RandomNormal',activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=128,kernel_initializer='RandomNormal',activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1,kernel_initializer='RandomNormal',activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.summary()
    
    train_history = model.fit(x=X_train, y=y_train, validation_split=0.1,epochs=100, batch_size=256, verbose=1)
    model.save('GC_DeepDTI.h5')
    
    score, acc = model.evaluate(X_test, y_test, verbose = 0)
    print('test accuracy:', acc)
    
    return train_history
    
# data label1 preprocessing
Data_prerpocessing_label1()
filename = 'data_label_1.pickle'
cluster_num = 100
label_1_num = Oversampling(filename, cluster_num)
print('data_label_1 finished')

# data label0 preprocessing
Data_prerpocessing_label0(label_1_num)
print('data_label_0 finished')

# concatenate two data
filename_1 = 'LABEL_ONE_OVERSAMPLE_cluster100.pickle'
filename_2 = 'Label_zeros'+str(label_1_num)+'.pickle'
Data_concatenate_PCA(filename_1, filename_2)
print('data concatenation finished')

# train model DNN
filename_3 = 'Data_preprocessed.pickle'
X_train, y_train, X_test, y_test = train_model(filename_3)
print('train DNN finished')

