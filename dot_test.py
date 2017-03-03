from __future__ import print_function
from keras.layers import Input, merge
from keras.models import Model
from keras import backend as K
import numpy as np

input_a = np.reshape([1, 2, 3, 4, 5, 6], (1, 3, 2))
# input_b = np.reshape([7, 1, 2, 0], (1, 2, 2))
input_b = np.reshape([-5, -5, 7, 7], (1, 2, 2))
input_c = np.reshape([1, 0, 2, 4], (1, 2, 2))

a = Input(shape=(3, 2))
b = Input(shape=(2, 2))
c = Input(shape=(2, 2))

# t1 = K.dot(a[3], b[3])
# print(t1)

def dim_sum(x, y):
    print(K.int_shape(x))
    print(K.int_shape(y))
    # t1 = K.dot(x, K.transpose(y))
    # K.reshape(t1, (3, 2))
    # t2 = K.sum(x, axis=1)
    # print(K.int_shape(x))
    # return K.batch_dot(x, y, axes=[2, 2])
    return K.batch_dot(x, y)
    # return K.dot(x, y)
    # return t2

def dim_sum_output_shape(x_shape, y_shape):
    return tuple(shape)

# concat = merge([a, b], mode='sum', dot_axes=0)
# dot = merge([b, c], mode='mul')
dot = merge([b, c], mode=lambda x: dim_sum(x[0], x[1]), output_shape=(2,2))
# cos = merge([a, b], mode='cos', dot_axes=1)

# model_concat = Model(input=[a, b], output=concat)
model_dot = Model(input=[b, c], output=dot)
# model_cos = Model(input=[a, b], output=cos)


# print(model_concat.predict([input_a, input_b]))
print(model_dot.predict([input_b, input_c]))
# print(model_cos.predict([input_a, input_b]))

print(input_b)
print()
print(input_c)
print()

