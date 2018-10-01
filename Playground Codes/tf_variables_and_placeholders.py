import tensorflow as tf


#Creating a tensor of shape 4 x 4 with maximum value of 1 and minimum of 0
my_tensor = tf.random_uniform((4,4),0,1)
print(my_tensor)

#Creating a variable
my_var = tf.Variable(initial_value = my_tensor)
print(my_var)

#All global variables must be initialized 

#initializing the variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(my_var))


#Declaring a placeholder
ph = tf.placeholder(tf.float32, shape=(None,5))