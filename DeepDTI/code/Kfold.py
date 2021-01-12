#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:35:47 2019

@author: shanjuyeh
"""
import pickle
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import regularizers
from sklearn.model_selection import train_test_split
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

#with open('/Users/shanjuyeh/Desktop/Project/GC_drug discovery/GC_DL predata/Data_preprocessed.pickle','rb') as file:
#    data = pickle.load(file)
#X = data.Data
#Y = data.Label
#X = np.stack(X)
#Y = np.stack(Y)

with open('/Users/shanjuyeh/Desktop/Project/GC_DeepDTI/GC_DL predata/drug_target_1.pickle', 'rb') as f:
    data_1 = pickle.load(f)
with open('/Users/shanjuyeh/Desktop/Project/GC_DeepDTI/GC_DL predata/drug_target0.pickle', 'rb') as f:
    data_0 = pickle.load(f)

data_0 = data_0.sample(n=80291, random_state=1)
data_0 = data_0.reset_index(drop=True)

data_1 = np.concatenate([data_1.iloc[i,0].tolist() for i in range(len(data_1))])
data_0 = np.concatenate([data_0.iloc[i,0].tolist() for i in range(len(data_0))])

drug_target = np.concatenate((data_1, data_0), axis=0)
label = np.concatenate((np.ones(data_1.shape[0]),np.zeros(data_0.shape[0])))

del data_1
del data_0

X_train, X_test, Y_train, Y_test = train_test_split(drug_target, label, test_size=0.3, random_state=0)


### Standard scaler
scaler = StandardScaler()
scaler.fit(X_train)
XX_train = scaler.transform(X_train)
XX_test = scaler.transform(X_test)

### PCA transform
pca_x  = PCA(n_components=1000)
pca_x.fit(XX_train)
x_train = pca_x.transform(XX_train)
x_test = pca_x.transform(XX_test)

# fix random seed for reproducibility
from numpy.random import seed
seed(7)

# define 5-fold cross validation test harness (USE THIS!!!!!!)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i=1

cvscores=[]
val_acc=[]
val_loss=[]
val_auc=[]

train_acc=[]
train_loss=[]
train_auc=[]

test_acc=[]
test_loss=[]

validate_acc=[]
validate_loss=[]

save=[]
for train_index, test_index in kfold.split(x_train, Y_train):
    
    X_trn, X_val = x_train[train_index], x_train[test_index]
    y_trn, y_val = Y_train[train_index], Y_train[test_index]
    
    save.append(y_trn)
    
    model=Sequential()
    model.add(Dense(units=512,input_dim=1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1,activation='sigmoid'))
    
    opt = optimizers.Adam(learning_rate=0.003)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)
    model.summary()
    
    #train_history = model.fit(np.stack(X[train]), np.stack(Y[train]), validation_split=0.1, epochs=100, batch_size=100, verbose=1, callbacks=[es])
    train_history = model.fit(X_trn, y_trn, validation_data = (X_val, y_val), epochs=100, batch_size=100, verbose=1)
    
    train_loss.append(train_history.history['loss'])
    train_acc.append(train_history.history['accuracy'])
    train_auc.append(train_history.history['auc'])
    val_loss.append(train_history.history['val_loss'])
    val_acc.append(train_history.history['val_accuracy'])
    val_auc.append(train_history.history['val_auc'])
    

    pre = model.predict(X_val)
    fpr,tpr,threshold = roc_curve(y_val, pre)
    
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
    
    results = model.evaluate(X_val, y_val)
    results = dict(zip(model.metrics_names,results))
    print('val loss:', results['loss'], 'val accuracy:', results['accuracy'])
    validate_acc.append(results['accuracy'])
    validate_loss.append(results['loss'])
    
    
    
    i= i+1

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'navy')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.3f )' % (mean_auc),lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('ROC_result_kfold.png')
plt.show()

print("Accuracy_validation_mean_5fold: %.6f ; Accuracy_validation_std_5fold: %.6f" % (np.mean(validate_acc), np.std(validate_acc)))
print("Loss_validation_mean_5fold: %.6f ; Loss_validation_std_5fold: %.6f" % (np.mean(validate_loss), np.std(validate_loss)))


min_num = 150
for i in train_loss:
    if len(i) < min_num:
        min_num = len(i)
####=========================plot
markersize = 2.75
plt.figure() 
for i, data in enumerate(train_loss, start=1):
    plt.plot(np.arange(min_num), data[:min_num], 'x--', label='train loss_'+str(i), markersize=markersize)
for i, data in enumerate(val_loss, start=1):
    plt.plot(np.arange(min_num), data[:min_num], 'o--', label='val loss_'+str(i), markersize=markersize)
train_loss_mean = np.sum([item[:min_num] for item in (train_loss)], axis=0)/len(train_loss)
val_loss_mean = np.sum([item[:min_num] for item in (val_loss)], axis=0)/len(val_loss)
plt.plot(np.arange(min_num), train_loss_mean, 'r', label='train loss mean', linewidth=3)
plt.plot(np.arange(min_num), val_loss_mean, 'b',  label='val loss mean', linewidth=3)
plt.legend(bbox_to_anchor=(1, 1.05), fontsize='small')
plt.title('Training and validation loss (5-fold cross validation)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('Training and validation loss.png',
            dpi = 200,
            format='png', 
            bbox_inches='tight')

plt.figure()
for i, data in enumerate(train_acc, start=1):
    plt.plot(np.arange(min_num), data[:min_num], 'x--', label='train accuracy_'+str(i), markersize=markersize)
for i, data in enumerate(val_acc, start=1):
    plt.plot(np.arange(min_num), data[:min_num], 'o--', label='val accuracy_'+str(i), markersize=markersize)
train_acc_mean = np.sum([item[:min_num] for item in (train_acc)], axis=0)/len(train_acc)
val_acc_mean = np.sum([item[:min_num] for item in (val_acc)], axis=0)/len(val_acc)
plt.plot(np.arange(min_num), train_acc_mean, 'r', label='train acc mean', linewidth=3)
plt.plot(np.arange(min_num), val_acc_mean, 'b', label='val acc mean', linewidth=3)
plt.legend(bbox_to_anchor=(1, 1.05), fontsize='small')
plt.title('Training and validation accuracy (5-fold cross validation)')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('Training and validation accuracy.png', 
            dpi = 200,
            format='png', 
            bbox_inches='tight')








## define 10-fold cross validation test harness
#seed =7
#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#
#cvscores = []
#tprs = []
#aucs = []
#mean_fpr = np.linspace(0,1,100)
#i=1
#
#
#for train, test in kfold.split(X, Y):
#    model=Sequential()
#    model.add(Dense(units=512,input_dim=1000,activation='relu'))
#    model.add(Dense(units=256,activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=128,activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=64,activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(units=1,activation='sigmoid'))
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
#    model.summary()
#    
#    #train_history = model.fit(np.stack(X[train]), np.stack(Y[train]), validation_split=0.1, epochs=10, batch_size=100, verbose=1 )
#    #loss, acc = model.evaluate(np.stack(X[test]), np.stack(Y[test]), verbose = 0)
#    
#    history = model.fit(np.stack(X[train]), np.stack(Y[train]), validation_split=0.1, epochs=100, batch_size=100, verbose=1 )
#    
#    scores = model.evaluate(np.stack(X[test]), np.stack(Y[test]), verbose=0)
#    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#    cvscores.append(scores[1] * 100)
#    
#    pre = model.predict(np.stack(X[test]))
#        
#    fpr,tpr,threshold = roc_curve(np.stack(Y[test]), pre)
#    tprs.append(interp(mean_fpr, fpr, tpr))
#    roc_auc = auc(fpr, tpr)
#    aucs.append(roc_auc)
#    plt.figure(1)
#    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#    
#    # plot loss during training
#    plt.figure(2)
#    plt.title('Loss')
#    plt.plot(history.history['loss'], label='train loss_%s'%i)
#    plt.plot(history.history['val_loss'], label='test loss_%s'%i)
#    plt.legend()
#    # plot accuracy during training
#    plt.figure(3)
#    plt.title('Accuracy')
#    plt.plot(history.history['accuracy'], label='train accuracy_%s'%i)
#    plt.plot(history.history['val_accuracy'], label='test accuracy_%s'%i)
#    plt.legend()
#    i= i+1
#
#print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'navy')
#mean_tpr = np.mean(tprs, axis=0)
#mean_auc = auc(mean_fpr, mean_tpr)
#plt.plot(mean_fpr, mean_tpr, color='blue',
#         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC')
#plt.legend(loc="lower right")
#plt.show()



