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