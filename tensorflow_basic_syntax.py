import tensorflow as tf

hello = tf.constant("Hello ")
world = tf.constant("World")

print(type(hello))
print(hello)

with tf.Session() as session:
    result = session.run(hello + world)

print(result)

a = tf.constant(10)
b = tf.constant(20)

print(type(a))

print( a + b )
print( a + b )

with tf.Session() as session:
    result_2 = session.run(a + b)
    print(type(result_2))
    print(result_2)

const = tf.constant(10)
fill_mat = tf.fill(((4,4)), 10)
my_zeros = tf.zeros((4,4))
my_ones = tf.ones((4,4))
my_random_n = tf.random_normal((4,4), mean = 0, stddev = 1.0)
my_random_u = tf.random_uniform((4,4), minval = 0, maxval = 1)

my_ops = [const, fill_mat, my_zeros, my_ones, my_random_n, my_random_u]

for operation in my_ops:
    with tf.Session() as session:
        print(session.run(operation))
    print('\n')