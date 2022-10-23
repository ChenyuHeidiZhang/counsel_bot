import tensorflow as tf 
import os
# tf.debugging.set_log_device_placement(True) 


# Explicitly place tensors on the DirectML device 

with tf.device('/GPU:0'): 

  a = tf.constant([1.0, 2.0, 3.0]) 

  b = tf.constant([4.0, 5.0, 6.0]) 



c = tf.add(a, b) 

print(c)