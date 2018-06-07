# Python
import six
print(six.__version__)
print(six.__file__)

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
