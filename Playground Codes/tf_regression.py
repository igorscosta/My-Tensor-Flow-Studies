import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 

#Generating some random data

x_data = np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns = ['X Data'])
y_df = pd.DataFrame(data=y_true, columns = ['Y'])


my_data = pd.concat([x_df, y_df], axis = 1)
print(my_data.head())

#Ploting some sample data

my_data.sample(n = 250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
plt.show()

# A million entries is too much for a Neural Network so..
#Creating some batches of data
batch_size = 8 

#Declaring some variables
m = tf.Variable(0.81)
b = tf.Variable(0.17)

#Declaring some placeholders
xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m*xph + b

#Loss Function
error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

#initializing the variables
init = tf.global_variables_initializer()

#creating a TF session
with tf.Session() as sess:
    sess.run(init)
    batches = 1000

    for i in range(batches):

        rand_ind = np.random.randint(len(x_data), size = batch_size)

        feed = {xph : x_data[rand_ind], yph : y_true[rand_ind]}

        sess.run(train, feed_dict = feed)

    model_m, model_b = sess.run([m, b])

y_hat = x_data * model_m + model_b
my_data.sample(250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
plt.plot(x_data, y_hat, 'r')
plt.show()


########################## Using TensorFlow Estimator API ######################

#Creating the features columns list
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]

#Estimator
estimator = tf.estimator.LinearRegressor(feature_columns = feat_cols)

from sklearn.model_selection import train_test_split

#Setting the train/test split with 70% training and 30% eval.
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size = 0.3, random_state = 101)
print(x_train.shape())

#Setting the estimator inputs. 

#Using Numpy (Also could be used with Pandas)
input_func = tf.estimator.inputs.numpy_input_fn({'x' : x_train}, y_train, batch_size = 8, num_epochs = None, shuffle = True)

