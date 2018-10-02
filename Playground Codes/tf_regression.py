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

#Ploting a sample data

my_data.sample(n = 250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
plt.show()


