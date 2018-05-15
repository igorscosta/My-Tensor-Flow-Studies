import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


######### Creating Data #############

x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))

print(x_data)
print('\n')

# y = mx + b + noise_levels
b = 5

y_true = (0.5 * x_data) + 5 + noise

my_data = pd.concat([pd.DataFrame(data = x_data, columns=['X Data']), pd.DataFrame(data = y_true, columns=['Y'])], axis = 1)
print(my_data.head())

my_data.sample(n = 250).plot(kind = 'scatter', x='X Data', y = 'Y')
plt.show()


# We will take the data in batches (1,000,000 points is a lot to pass in at once)

batch_size = 8

# Variables

m = tf.Variable(0.81)
b = tf.Variable(0.17)

# Placeholders

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

# Graph

y_model = m * xph + b

# Loss Function 

error = tf.reduce_sum(tf.square(yph - y_model))

# Optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)

# Initialize Variables

init = tf.global_variables_initializer()

# Sessions

with tf.Session() as sess:

    sess.run(init)

    batches = 1000

    for i in range(batches):

        rand_ind = np.random.randint(len(x_data), size = batch_size)
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        sess.run(train,feed_dict=feed)
        
    model_m,model_b = sess.run([m,b])

    print(model_m)
    print(model_b)

# Results

y_hat = x_data * model_m + model_b

my_data.sample(n = 250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
plt.plot(x_data, y_hat, 'r')
plt.show()


# tf.estimator API

feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]

# Train Test Split