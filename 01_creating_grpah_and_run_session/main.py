import tensorflow as tf

## creating a computatation graph
x = tf.Variable(3, name= "x")
y = tf.Variable(4, name= "x")
f = x * x + y * y

sess = tf.Session()

sess.run(x.initializer)
sess.run(y.initializer)

result = sess.run(f)

print result

sess.close()
