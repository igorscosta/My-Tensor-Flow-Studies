import tensorflow as tf 

n1 = tf.constant(1)
n2 = tf.constant(2)

n3 = n1 + n2

# Using with auto-closes the session
with tf.Session() as sess:
    result = sess.run(n3)
    print(result)

# When you start TF, a default Graph is created, you can create additional graphs easily:
print(tf.get_default_graph())

g = tf.Graph()
print(g)

#Setting a graph as the default:

graph_one = tf.get_default_graph()
graph_two = tf.Graph()

print(graph_one is tf.get_default_graph())
print(graph_two is tf.get_default_graph())

#Setting graph_two as default
with graph_two.as_default():
    print(graph_two is tf.get_default_graph())

#Graph two is not global default 
print(graph_two is tf.get_default_graph())