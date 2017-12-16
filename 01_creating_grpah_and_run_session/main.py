import tensorflow as tf

## creating a computatation graph
x = tf.Variable(3, name= "x")
y = tf.Variable(4, name= "x")
f = x * x + y * y


#
# method 1
#
print "method 1"
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print result
sess.close()




#
# method 2
#
print "method 2"
with tf.Session() as sess:
	x.initializer.run()
	y.initializer.run()
	result = f.eval()
print result



#
# method 3
#
print "method 3"

init = tf.global_variables_initializer()

with tf.Session() as sess:
	init.run() # init all the variables
	result = f.eval()
print result
