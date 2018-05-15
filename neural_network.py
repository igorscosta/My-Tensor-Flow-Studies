import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

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

print('\n')

################ Simple Regression Example ##################

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
print(x_data)
print('\n')

y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
print(y_label)

plt.plot(x_data, y_label, '*')
plt.show()

######## y = mx + b #########

m = tf.Variable(0.44)
b = tf.Variable(0.87)

error = 0

for x,y in zip(x_data, y_label):

    y_hat = m * x + b

    error += (y - y_hat) ** 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    training_steps = 100

    for i in range(training_steps):
        sess.run(train)

    final_slope, final_intercept = sess.run([m, b])

x_test = np.linspace(-1,11,10)

# y = mx + b
y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()