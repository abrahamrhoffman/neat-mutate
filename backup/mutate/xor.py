from numpy.random import randn as rnd
import tensorflow as tf
import numpy as np
#from multigpu import MultiGPU

def main():
    ### Data ###
    # Raw Data
    x = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]], dtype=np.float32)

    y = np.array([[1,0,0,1]], dtype=np.float32).T

    # Preprocess Data
    # - Not Needed -

    # Tensorflow Graph Data
    X = tf.constant(x)
    Y = tf.constant(y)

    x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

    Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta1")
    Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Theta2")

    Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
    Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")

    A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
    Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

    cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)

    train_step = tf.train.GradientDescentOptimizer(1).minimize(cost)

    XOR_X = [[0,0],[0,1],[1,0],[1,1]]
    XOR_Y = [[0],[1],[1],[0]]

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000000):
        sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
        if i % 1000 == 0:
            print('-------------------------------')
            print('Epoch ', i)
            print('-------------------------------')
            print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
            print('Theta1 ', sess.run(Theta1))
            print('Bias1 ', sess.run(Bias1))
            print('Theta2 ', sess.run(Theta2))
            print('Bias2 ', sess.run(Bias2))
            print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
            print('-------------------------------')

    ### Neuroevolution ###
    # Hyperparameters

    # Model Instantiation
    #model = tf.multiply(X,Y)

    ### Execution ###
    # GPU Computation
    #distribute = MultiGPU(3)

    # Define CPU Ops
    #ops = ['tf.add_n(self.tower)']

    # Execute
    #result = distribute.run(model)#, ops)
    #for r in result:
    #    print r

if __name__ == "__main__":
    main()
