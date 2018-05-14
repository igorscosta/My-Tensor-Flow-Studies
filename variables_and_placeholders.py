import tensorflow as tf

sess = tf.Session()
my_tensor = tf.random_uniform((4,4),0,1)
print(my_tensor)

my_var = tf.Variable(initial_value=my_tensor)
print(my_var)
