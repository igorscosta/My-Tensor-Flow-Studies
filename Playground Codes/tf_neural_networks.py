import numpy as np
import tensorflow as tf 

#Generating some random seeds with NP and Tensor Flow

np.random.seed(101)
tf.set_random_seed(101)

#Generating some random data
rand_a = np.random.uniform(0,100,(5,5))
print(rand_a)

rand_b = np.random.uniform(0,100,(5,1))
print(rand_b)

#Creating some placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#Defining some operations
add_op = a + b
mul_op = a * b

# Feeding a dict with placeholders and variables and running an operation 
with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={a:rand_a, b:rand_b})
    print(add_result)

    mult_result = sess.run(mul_op, feed_dict={a:rand_a, b:rand_b})
    print(mult_result)


print('\n')
print('RUNING A EXAMPLE OF A VERY SIMPLE NEURAL NETWORK')
print('\n')


#Example Neural Network 

n_features = 10
n_dense_neurons = 3

#Defining a placeholder 
x = tf.placeholder(tf.float32, (None, n_features))

#Declaring variables 
#Weight with the shape of number of features by number of dense neurons 
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))

#Declaring some operations 
xW = tf.matmul(x,W)
z = tf.add(xW,b)

#Declaring the activation function 
a = tf.sigmoid(z)

#Initializing the variables
init = tf.global_variables_initializer()

#Initilizaing a session
with tf.Session() as sess:
    sess.run(init)

    layer_out = sess.run(a, feed_dict={x:np.random.random([1, n_features])})

print('LAYER OUT = ', layer_out)