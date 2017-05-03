#!/usr/bin/python
from __future__ import print_function
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np

def create_data():
    '''
    XOR Data.
    '''
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    return X,Y

def genome(data):
    '''
    A Neural Network Definition.
    '''
    X,Y = data

    x_ = tf.placeholder(tf.float32, shape=[X.shape[0],X.shape[1]]) # Input Nodes: Layer 0
    y_ = tf.placeholder(tf.float32, shape=[Y.shape[0],Y.shape[1]]) # Output Nodes: Output Layer

    return x_,y_

def phenome(agenome):
    '''
    A Neural Network Constructor: Accepts a Genome as input.
    '''
    x_,y_ = agenome

    w1 = tf.Variable(tf.random_uniform([2,1], -1, 1)) # Weights for all nodes in Layer 1
    b1 = tf.Variable(tf.zeros([1]))                   # Bias for all nodes in Layer 1
    op1 = tf.matmul(x_, w1) + b1                      # Construct the Operation for Layer 1
    a1 = tf.sigmoid(op1)                              # Activation Function applied to the Operation

    return w1,b1,op1,a1

def fitness():
    pass

def evaluate(data,agenome,aphenome):
    X,Y = data
    x_,y_ = agenome
    w1,b1,op1,a1 = aphenome

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    result = sess.run(a1, feed_dict={x_: X, y_: Y})

    writer = tf.summary.FileWriter("./logs", sess.graph)

    print('Epoch 0:')
    print()
    print('Expected Output:')
    print('----------------')
    print(Y)
    print('Evolved Model:')
    print('----------------')
    print(result)


def main():
    data = create_data()
    agenome = genome(data)
    aphenome = phenome(agenome)
    evaluate(data,agenome,aphenome)

if __name__ == "__main__":
    main()

