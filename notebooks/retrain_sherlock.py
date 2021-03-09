# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:30:56 2021

@author: 459312
"""
import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import tensorflow as tf
import time
import glob 

from sherlock import helpers
from sherlock.features.preprocessing import extract_features, convert_string_lists_to_lists, prepare_feature_extraction 
from sherlock.features.preprocessing import extract_features_chars, extract_features_embed, extract_features_words, extract_features_paras
from sherlock.deploy.train_sherlock import train_sherlock
from sherlock.deploy.predict_sherlock import predict_sherlock
from memory_profiler import profile

import warnings
warnings.filterwarnings("ignore")

@profile
def load_transdata(train_test):
    if train_test == 'train':
        datapaths = glob.glob(r'D:\sherlock-project\notebooks\transfered_dataframe\X_train*')
    elif train_test == 'test':
        datapaths = glob.glob(r'D:\sherlock-project\notebooks\transfered_dataframe\X_test*')
    elif train_test == 'valid':
        datapaths = glob.glob(r'D:\sherlock-project\notebooks\transfered_dataframe\X_valid*')
        
    dataframes = [pd.read_parquet(data) for data in datapaths]
    df_op = pd.DataFrame()
    for i in range(len(datapaths)):
        df_op = df_op.append(dataframes[i])
    df_op.reset_index(drop =True,inplace =True)
    return df_op

@profile
def feature_extract(train_samples_converted,val_samples_converted,test_samples_converted):
    starttime = time.time()
    pool1 = mp.Pool()
    X_train_res = pool1.map(extract_features, train_samples_converted)
    pool1.close()
    pool1.join()
    
    pool2 = mp.Pool()
    X_val_res = pool2.map(extract_features, val_samples_converted)
    pool2.close()
    pool2.join()
    
    pool3 = mp.Pool()
    X_test_res = pool3.map(extract_features, test_samples_converted)
    pool3.close()
    pool3.join()
    
    # 另外一種多進程的方法
    #multi_res = [pool.apply_async(extract_features, (data[i],)) for i in range(n_values)]
    #res = [res.get() for res in multi_res]
    temptime = time.time()
    print('That took {} seconds'.format(temptime - starttime)) # 平均一筆資料要花1.7秒


    column_names = X_train_res[0].keys()
    X_train_res = [list(x.iloc[0,:]) for x in X_train_res]
    X_train_res = pd.DataFrame(X_train_res,columns=column_names)
    
    X_val_res = [list(x.iloc[0,:]) for x in X_val_res]
    X_val_res = pd.DataFrame(X_val_res,columns=column_names)  
    
    X_test_res = [list(x.iloc[0,:]) for x in X_test_res]
    X_test_res = pd.DataFrame(X_test_res,columns=column_names)  
    print('That took {} seconds'.format(time.time() - temptime)) 
    
    # 補空值
    #train_columns_means = pd.DataFrame(X_train_res.mean()).transpose()
    #X_train_res.fillna(train_columns_means.iloc[0], inplace=True)
    #X_val_res.fillna(train_columns_means.iloc[0], inplace=True)
    #X_test_res.fillna(train_columns_means.iloc[0], inplace=True)
    
    return X_train_res,X_val_res,X_test_res

def data_preprocess(train_test,sampling = False):
    train_samples = pd.read_parquet(f'../data/data/raw/{train_test}_values.parquet')
    train_labels = pd.read_parquet(f'../data/data/raw/{train_test}_labels.parquet')
    
    train_samples.reset_index(drop = True,inplace=True)
    train_labels.reset_index(drop = True,inplace=True)
    
    if sampling == True:
        unique_labels = np.unique(train_labels)
        sampling_id = []
        for lb in unique_labels:
            sampling_id.append(train_labels[train_labels['type']==lb].index[0])
        train_samples = train_samples.iloc[sampling_id,:]
        train_labels = train_labels.iloc[sampling_id,:]
    return train_samples,train_labels

if __name__ == '__main__':
    
    train_samples,train_labels = data_preprocess('train',sampling = False)
    validation_samples,validation_labels = data_preprocess('val',sampling = False)
    test_samples,test_labels = data_preprocess('test',sampling = False)
    
    
    train_samples_converted, y_train = convert_string_lists_to_lists(train_samples.iloc[160000:204000,:], train_labels.iloc[160000:204000,:], "values", "type")
    val_samples_converted, y_val = convert_string_lists_to_lists(validation_samples.iloc[1:5,:], validation_labels.iloc[1:5,:], "values", "type")
    test_samples_converted, y_test = convert_string_lists_to_lists(test_samples.iloc[50000:68000,:], test_labels.iloc[50000:68000,:], "values", "type")
    
    del train_samples
    del validation_samples
    del test_samples
    
    train_samples_converted = [train_samples_converted[i:i+1] for i in range(len(train_samples_converted))]
    val_samples_converted = [val_samples_converted[i:i+1] for i in range(len(val_samples_converted))]
    test_samples_converted = [test_samples_converted[i:i+1] for i in range(len(test_samples_converted))]
    print("data preprocess done")
    
    X_train,X_val,X_test = feature_extract(train_samples_converted,val_samples_converted,test_samples_converted)
    del train_samples_converted
    del val_samples_converted
    del test_samples_converted
    print("feature extraction done")
    """
    X_train, X_test = load_transdata('train'), load_transdata('test')
    
    train_sherlock(X_train, y_train, X_train, y_train, nn_id='retrained_sherlock')
    print('Trained and saved new model.')


    predicted_labels = predict_sherlock(X_test, nn_id='retrained_sherlock')
    f1 = f1_score(y_test, predicted_labels,average='weighted')
    print("f1 score: ",f1)
    """
    
    