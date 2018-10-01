import tensorflow as tf 

a = tf.constant(10)
b = tf.constant(20)

#Tensor Flow tracks the additions in Background

print(a + b) #add 0
print(a + b) #add 1

with tf.Session() as sess:
    result = sess.run(a + b)
    print(result)

