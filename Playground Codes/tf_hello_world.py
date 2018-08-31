import tensorflow as tf 


print(tf.__version__)

# First hello world with tensors! <3

hello = tf.constant('Hello ')
print(type(hello))

world = tf.constant('World!')
result = hello + world
print(result)

# Operations on Tensor Flow must be executed on a Session

with tf.Session() as sess:
    result = sess.run(hello + world)

print(result)