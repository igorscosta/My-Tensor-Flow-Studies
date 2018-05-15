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




