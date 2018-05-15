import numpy as np
import tensorflow as tf 

np.random.seed(101)
tf.set_random_seed(101)

rand_a = np.random.uniform(0,100,(5,5))
print(rand_a)

rand_b = np.random.uniform(0,100,(5,1))
print(rand_b)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a + b
mul_op = a * b

with tf.Session() as sess:
    add_result = sess.run(add_op,feed_dict={a:rand_a, b:rand_b})
    print(add_result)
    print('\n')
    mult_result = sess.run(mul_op,feed_dict={a:rand_a, b:rand_b})
    print(mult_result)


############## Example Neural Network ####################

n_features = 10
n_dense_neurons = 3
 
x = tf.placeholder(tf.float32, (None, n_features))
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))
xW = tf.matmul(x,W)
z = tf.add(xW,b)
a = tf.sigmoid(z)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    layer_out = sess.run(a, feed_dict={x:np.random.random([1, n_features])})

    print(layer_out)