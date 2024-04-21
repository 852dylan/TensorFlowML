
from __future__ import print_function
import tensorflow as tf
#tensor constants

tensorObject = tf.constant(2)
tensorObject2 = tf.constant(5)
mult = tf.multiply(tensorObject,tensorObject2)
print(mult)

#matrix multiplications
matrix1 = tf.constant([[1.,2.],[3.,4.]])
matrix2 = tf.constant([[4.,5.],[1.,2.]])
product = tf.matmul(matrix1,matrix2)
print("Type of Product:",type(product))
print("Print Product: ",product)
print("product by itself:")
product

# numpy is very important when working with tensor flow
