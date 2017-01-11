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

2.Run:   
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
corss_entropy = tf.reduce  ... tbc
```
