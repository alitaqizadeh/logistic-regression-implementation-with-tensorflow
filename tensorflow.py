# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def logisticRegression(self, number_of_features, x_train_data, y_train_data, x_test_data, y_test_data):
    
    learning_rate = 0.01
    training_epochs = 1000

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, number_of_features], name="X")
    Y = tf.placeholder(tf.float32, [None, 1], name="Y")

    # Initialize our weigts & bias
    W = tf.get_variable("W", [number_of_features, 1], initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [1], initializer = tf.zeros_initializer())

    Z = tf.add(tf.matmul(X, W), b)
    prediction = tf.nn.sigmoid(Z)

    # Calculate the cost
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))

    # Use Adam as optimization method
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    cost_history = np.empty(shape=[1],dtype=float)

    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(training_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={X: x_train_data, Y: y_train_data})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
            cost_history = np.append(cost_history, c)
            
            
        # Calculate the correct predictions
        correct_prediction = tf.to_float(tf.greater(prediction, 0.5))
    
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(Y, correct_prediction)))
    
        print ("Train Accuracy:", accuracy.eval({X: x_train_data, Y: y_train_data}))
        print ("Test Accuracy:", accuracy.eval({X: x_test_data, Y: y_test_data}))