import tensorflow as tf
import pandas as pd
import numpy as np

def read_data():
    """read data"""
    iris_data = pd.read_csv("/home/prakash/anaconda3/lib/python3.5/site-packages/tensorflow/contrib/learn/python/learn/datasets/data/iris.csv",\
                            header=None,index_col=False, names =['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'labels'])
    
    #assign some varibles to store info of dataset
    no_of_datapoints = iris_data.ix[0, 0].astype(np.float32)
    no_of_features = iris_data.ix[0, 1].astype(np.float32)
    labels = iris_data.ix[0, 2:]
    
    iris_data.drop(iris_data.index[[0]],inplace=True)
    iris_data.reset_index(inplace=True,drop=True)
#    print("iris_data :\n %r" % iris_data.head())
    return iris_data, no_of_datapoints, int(no_of_features), labels
    
def add_vector_labels(iris_data):
    """add labels in vector form"""
    
    a = np.asarray([1,0,0])
    b = np.asarray([0,1,0])
    c = np.asarray([0,0,1])
    iris_data['y'] = iris_data['labels'].map({ '0':a,'1':b,'2':c})
    return iris_data
    
def neural_network(x, weights, biases):
    """function to define the neural network"""

    #layer 1 computation
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)
    
    #layer2 computation
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)
    
    #output layer computation
    out_layer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    out_layer = tf.nn.softmax(out_layer)
    return out_layer
    
def main():
    
    iris_data, no_of_datapoints, no_of_features, labels = read_data()
    no_of_outputs = len(labels)
    print("no_of_features",no_of_features)
    print("no_of_outputs",no_of_outputs)
    iris_data = add_vector_labels(iris_data)
    
    """prepare data for tensorflow graph"""

    col = [col for col in iris_data.columns if col not in ['labels', 'y']]
    print("col:%r"%col)
#    inputX = iris_data[col].as_matrix()
#    print(inputX.shape)
#    inputY = iris_data['y'].as_matrix()
#    print(inputY)
    
    #shuffle data
    data = iris_data.iloc[np.random.permutation(len(iris_data))]
                          
    #data reset index after shuffling
    data.reset_index(inplace=True, drop=True)
 
    #create input and output variable
    inputX = iris_data.ix[0:130,col].as_matrix()
    inputY = iris_data.ix[0:130,'y'].as_matrix()
    
    #test set input and output
    input_testX = iris_data.ix[131:,col].as_matrix()
    input_testY = iris_data.ix[131:,'y'].as_matrix()
    
    """hyper-parameters"""
    
    learning_rate = 0.003
    training_epochs = 3000
    
    display_steps = 500
    n_samples = inputY.size
    n_hidden_1 = 2 # 1st layer number of features
    n_hidden_2 = 2 # 2nd layer number of features
    
    """create out computation graph"""
    
    #tensors for input     
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
    cost = tf.reduce_sum(tf.pow(y_ - prediction,2))/(2*n_samples)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    
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
#    print(sess.run(y, feed_dict = {x:inputX}))
#    print(inputY)
    
if __name__ == "__main__":
    main()
    tf.reset_default_graph()
    
