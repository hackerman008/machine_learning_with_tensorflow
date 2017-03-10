"""neural network for MNIST dataset"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

def read_dataset():
    """function to load dataset"""
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(mnist.train.num_examples)
    return mnist
    
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
    
    #output layer computation
    out_layer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    #out_layer = tf.nn.relu(out_layer)
    return out_layer

def main():
    """main function"""
    
    mnist = read_dataset()
    no_of_features = 28*28
    no_of_outputs = 10
    
    
    """define hyper-parameters"""    
    learning_rate = 0.003
    training_epochs = 20000
    batch_size = 100
    n_hidden_1 = 250 # 1st layer number of features
    n_hidden_2 = 250 # 2nd layer number of features

    
    display_steps = 500
    
    
    x = tf.placeholder(tf.float32, [None, no_of_features])
    y_ = tf.placeholder(tf.float32, [None, no_of_outputs])
    
    #store layers weights and biases
    tf.set_random_seed(2)
    weights = {'h1':tf.Variable(tf.random_normal([no_of_features, n_hidden_1])),
                'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'out':tf.Variable(tf.random_normal([n_hidden_2, no_of_outputs]))
                }
    

    biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([no_of_outputs]))
                }
        
    prediction = neural_network(x, weights, biases)
    print(prediction)
    
    #cost function
    #cost = tf.reduce_sum(tf.pow(y_ - prediction,2),[0,1])/(2*n_samples)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
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
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        sess.run(optimizer, feed_dict ={x:batch_x,y_:batch_y})
        
        if i%display_steps == 0:
            cc = sess.run(cost, feed_dict={x:batch_x,y_:batch_y})
            
            print("training step:",i,"\t cost:%r"%cc)

    print("accuracy: %r"%sess.run(accuracy,feed_dict={x:mnist.test.images , y_:mnist.test.labels }))


    return

if __name__ == "__main__":

    main()
    tf.reset_default_graph()