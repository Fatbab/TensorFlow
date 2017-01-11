Some quick tips form [TensorFlow](https://www.tensorflow.org/get_started/os_setup) project website.
###Install TensorFlow:   
```
pip install tensorflow
```

###Run a basic handwriting classification:   
1. Find where tensorflow is installed on your machine:   
```
python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
```
That is when using Python2. If tensorflow is installed for Python3, refer to the main reference atop.

2. Run:   
```
python -m tensorflow.models.image.mnist.convolutional
```

It should starts loading the sample data and running the classification in 100-batch steps. But what's actually happening?

##Peeling off one layer
Here, we see how the above classification is actually done. The full description is available at the [TensorFlow](https://www.tensorflow.org/tutorials/mnist/beginners/) project webpage.
We will be implementing:  y = softmax(Wx+b)

```
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.variable(tf.zeros[784,10])
b = tf.variable(tf.zeros[10])
y = tf.nn.softmax(tf.matmul(x, W) + b)  ## the predicted labels
y_ = tf.placeholder(tf.float32, [None, 10])  ## the actual labels
corss_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```
At this point, we have created the computation graph and defined the components. We will now define the initializer, start a session and let the trainig run in batches of 100 samples:

```
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000)
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```
Now we'll define evaluation/test mechanism:
```
correct_prediction = tf.equal(tf.argmax(y_), tf.argmax(y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
```

###A couple of points:
1. Entropy: First, remember the Shannon Entropy from Information Theory defined as _the expected value of information contained in every bit/message_. See the mathematical representation [here.](https://en.wikipedia.org/wiki/Entropy_(information_theory)#Definition) 
2. Cross Entropy:
  Intuitively speaking, it is the log-liklihood for data y'_i under model y_i. [This page](http://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks) has a good walkthrough of how the formula is derived. 
  The notion of cross_entropy here works as our cost function. [This is a very nice tutorial on Cross Entropy.](http://neuralnetworksanddeeplearning.com/chap3.html)   

3. `tf.argmax()`: This function gives the index of highest entry in tensor along some axis. So `tf.argmax(y,1)` is the label that the model thinks is most likely for each input while `tf.argmax(y',1)` is the actual label.

4. `tf.equal` returns binary {0, 1} values and that makes defining accuracy as simple as means of the `tf.equal` output. 
