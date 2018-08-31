import tensorflow as tf 

const = tf.constant(10)
fill_mat = tf.fill((4,4), 10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandn = tf.random_normal((4,4), mean=0, stddev = 1.0)
myrandu = tf.random_uniform((4,4), minval=0, maxval=1)

my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu]

with tf.Session() as sess:
    for op in my_ops:
        print(sess.run(op))
        print('\n')

#Matrix multiplication

a = tf.constant([ [1,2],
                  [3,4] ])

#getting matrix shape
print(a.get_shape())

b = tf.constant([[10],[100]])
print(b.get_shape())

with tf.Session() as sess:
    result = tf.matmul(a,b)
    sess.run(result)
    print(result)
    print(sess.run(result))  
    print(result.eval())