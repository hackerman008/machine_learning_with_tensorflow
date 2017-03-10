#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:47:09 2017

@author: prakash
"""

import os
import pandas as pd
import numpy as np

#splitting module

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation as cval
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

import tensorflow as tf


#load data
def load_data():
    
    df=pd.read_csv('train.csv')
    return df

def create_column_title(df):
    df['Title']='Mr.'
    for i in range(0,891):    
        if 'Mr.' or 'Master.' or 'Mrs.' or 'Miss.' in df['Name'][i]:
            if 'Mr.' in df['Name'][i]:        
                df.ix[i,'Title']='Mr.'
            elif 'Master.' in df['Name'][i]:
                df.ix[i,'Title']='Master.'
            elif 'Mrs.' in df['Name'][i]:        
                df.ix[i,'Title']='Mrs.'
            elif 'Miss.' in df['Name'][i]:
                df.ix[i,'Title']='Miss.'
    return df

def create_column_deck(df):
    df['Deck']='U'
    df['Cabins_owned']=0
    for i in range(0,891):
        if pd.isnull(df.ix[i,'Cabin']) :
            df.ix[i,'Deck']='Unknown' 
        else :   
            df.ix[i,'Deck']=df.ix[i,'Cabin'][0]
            df.ix[i,'Cabins_owned']=len(df.ix[i,'Cabin'].split())
    
    return df
    
def create_column_totalfamilysize(df):
    df['total_family_size']=df['SibSp']+df['Parch']          
    return df
   
   
def wrangling_data(df):
    #wrangling data
    df=df.drop(axis=1,labels=['PassengerId','Name','Ticket','Cabin'])
    df['Embarked']=df['Embarked'].fillna(method='ffill')
    df['Age']=df['Age'].fillna(df.Age.median())
    
    #creating dummy columns
    df=pd.get_dummies(df,columns=['Sex','Embarked','Title','Deck'])
      
    return df

def add_vector_labels(df):
    """add labels in vector form"""
    
    a = np.asarray([1,0])
    b = np.asarray([0,1])
    df['y'] = df['Survived'].map({ 0:a,1:b})
    return df

def neural_network(x, weights, biases):
    """function to define the neural network"""

    #layer 1 computation
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.relu(layer1)
    print(layer1)
    
    #layer2 computation
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.relu(layer2)
    print(layer2)
    
    layer3 = tf.add(tf.matmul(layer2, weights['h3']), biases['b3'])
    layer3 = tf.nn.sigmoid(layer3)
    print(layer3)
    
    #output layer computation
    out_layer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    out_layer = tf.nn.softmax(out_layer)
    return out_layer   

def main():
    #loading data    
    df=load_data()
    print(df.shape)
    
    #feature engineering
    df=create_column_title(df)
    df=create_column_deck(df)
    df=create_column_totalfamilysize(df)
    
    #cleaning data    
    df=wrangling_data(df)
    print(df.shape)
    
    ### add function to made label "y" column
    df = add_vector_labels(df)
    
    #store columns to be used as input
    col = [col for col in df.columns if col not in ['Survived', 'y']]
    print("col:%r"%col)
    
    no_of_datapoints = df.shape[0]
    no_of_features = len(col)
    no_of_outputs = len(df['Survived'].unique())

    """shuffle data so the neural network learns the pattern in data better"""
#    np.random.seed(seed=123) #seed the random generator to reproduce the result
#    data = df.iloc[np.random.permutation(len(df))]
    data.reset_index(inplace=True, drop=True)

        
    #creating input and output variables
    temp = no_of_datapoints - 100
    inputX = data.ix[0:temp,col].as_matrix()
    inputY = data.ix[0:temp,'y'].as_matrix()
    
    #test set input and output
    input_testX = data.ix[temp+1:,col].as_matrix()
    input_testY = data.ix[temp+1:,'y'].as_matrix()

    """define hyper-parameters"""    
    learning_rate = 0.003
    training_epochs = 10000
    n_hidden_1 = 15 # 1st layer number of features
    n_hidden_2 = 15 # 2nd layer number of features
    n_hidden_3 = 20
    
    display_steps = 500
    n_samples = inputY.size
    
    """create out computation graph"""
    #tensors for input     
    x = tf.placeholder(tf.float32, [None, no_of_features])
    y_ = tf.placeholder(tf.float32, [None, no_of_outputs])
    
    #store layers weights and biases
    tf.set_random_seed(2)
    weights = {'h1':tf.Variable(tf.random_normal([no_of_features, n_hidden_1])),
                'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'h3':tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
                'out':tf.Variable(tf.random_normal([n_hidden_2, no_of_outputs]))
                }
    

    biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'b3': tf.Variable(tf.random_normal([n_hidden_3])),
                'out': tf.Variable(tf.random_normal([no_of_outputs]))
                }
        
    prediction = neural_network(x, weights, biases)
    print(prediction)
     
    #cost function
    cost = tf.reduce_sum(tf.pow(y_ - prediction,2),[0,1])/(2*n_samples)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    #cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction), reduction_indices=[1]))
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    """check accuracy"""
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    """initialize variables and start session"""
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(training_epochs):
        sess.run(optimizer, feed_dict ={x:inputX,y_:[t for t in inputY]})
        
        if i%display_steps == 0:
            cc = sess.run(cost, feed_dict={x:inputX,y_:[t for t in inputY]})
            
            print("training step:",i,"\t cost:%r"%cc)
            
    #check accuracy of model
    print("accuracy: %r"%sess.run(accuracy,feed_dict={x: input_testX, y_:[t for t in input_testY]}))
    
    """make prediction and compare it with the test set actual prediction"""
   
#    get_prediction = tf.argmax(prediction,1)
#    a=sess.run(get_prediction, feed_dict = {x:input_testX})
#    b = np.argmax([t for t in input_testY],1)    
#    for i in range(input_testY.size):
#        print("i:%r"%(i+1),"input:%r"%a[i],"predicted:%r"%b[i])


if __name__ == '__main__':
    main()
    tf.reset_default_graph()
    
