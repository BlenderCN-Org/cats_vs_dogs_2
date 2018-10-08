import tensorflow as tf
import tensorflow.contrib.slim as slim

def Stem(inputs):
    output=slim.conv2d(inputs,32,[3,3],stride=2,padding='VALID')
    output=slim.conv2d(output,32,[3,3],padding='VALID')
    output=slim.conv2d(output,64,[3,3])
    output_left=slim.max_pool2d(output,stride=2,[3,3])
    output_right=slim.conv2d(output,96,[3,3],stride=2,padding='VALID')
    output=tf.concat([output_left,output_right],3)

    output_left=slim.conv2d(output,64,[1,1])
    output_left=slim.conv2d(output_left,96,[3,3],padding='VALID')
    output_right=slim.conv2d(output, 64, [1, 1])
    output_right=slim.conv2d(output_right,64,[7,1])
    output_right=slim.conv2d(output_right,64,[1,7])
    output_right=slim.conv2d(output_right,96,[3,3],padding='VALID')
    output=tf.concat([output_left,output_right],3)

    output_left=slim.conv2d(output,192,[3,3],padding='VALID')
    output_right=slim.max_pool2d(output,stride=2,padding='VALID')
    output=tf.concat([output_left,output_right],3)

    return tf.nn.relu(output)

def Inception_ResNet_A(inputs):
    output_res=tf.identify(inputs)

    output_inception_a=slim.con2d(inputs,32,[1,1])
    output_inception_a=slim.conv2d(output_inception_a,384,[1,1],activation_fn=None)

    output_inception_b=slim.conv2d(inputs,32,[1,1])
    output_inception_b=slim.conv2d(output_inception_b,32,[3,3])
    output_inception_b=slim.conv2d(output_inception_b,384,[1,1],activation_fn=None)

    output_inception_c=slim.conv2d(inputs,32,[1,1])
    output_inception_c=slim.conv2d(output_inception_c,48,[3,3])
    output_inception_c=slim.conv2d(output_inception_c,64,[3,3])
    output_inception_c=slim.conv2d(output_inception_c,384,[1,1],activation_fn=None)

    output_inception=tf.add_n([output_inception_a,output_inception_b,output_inception_c])
    output_inception=tf.multiply(output_inception,0.1)
    output_inception=tf.add_n([output_inception,output_res])

    return tf.nn.relu(output_inception)

def Reduction_A(inputs):
    output_a=slim.max_pool2d(inputs,[3,3])

    output_b=slim.conv2d(inputs,384,[3,3],stride=2,padding='VALID')

    output_c=slim.conv2d(inputs,256,[1,1])
    output_c=slim.conv2d(output_c,256,[3,3])
    output_c=slim.conv2d(output_c,384,[3,3],stride=2,padding='VALID')

    output=tf.concat([output_a,output_b,output_c],3)

    return output

def Inception_ResNet_B(inputs,activation_fn=tf.nn.relu):
    output_res=tf.identify(inputs)
    output_a=slim.conv2d(inputs,192,[1,1])
    output_a=slim.conv2d(output_a,1152,[1,1],activation_fn=None)
    output_b = slim.conv2d(inputs, 128, [1, 1])
    output_b = slim.conv2d(output_b, 160, [1, 7])
    output_b = slim.conv2d(output_b, 192, [7, 1])
    output_b = slim.conv2d(output_b, 1152, [1, 1], activation_fn=None)
    output = tf.add_n([output_a, output_b])
    output = tf.multiply(output, 0.1)
    return tf.nn.relu(tf.add_n([output_res, output]))


def Reduction_B(inputs):
    output_a = slim.max_pool2d(inputs, [3, 3])

    output_b = slim.conv2d(inputs, 256, [1, 1])
    output_b = slim.conv2d(output_b, 384, [3, 3], stride=2, padding='VALID')

    output_c = slim.conv2d(inputs, 256, [1, 1])
    output_c = slim.conv2d(output_c, 256, [1, 1])
    output_c = slim.conv2d(output_c, 288, [3, 3], stride=2, padding='VALID')

    output_d = slim.conv2d(inputs, 256, [1, 1])
    output_d = slim.conv2d(output_d, 288, [3, 3])
    output_d = slim.conv2d(output_d, 320, [3, 3], stride=2, padding='VALID')

    return tf.nn.relu(tf.concat([output_a, output_b, output_c, output_d], 3))


def Inception_ResNet_C(inputs, activation_fn=tf.nn.relu):
    output_res = tf.identity(inputs)

    output_a = slim.conv2d(inputs, 192, [1, 1])
    output_a = slim.conv2d(output_a, 2144, [1, 1], activation_fn=None)

    output_b = slim.conv2d(inputs, 192, [1, 1])
    output_b = slim.conv2d(output_b, 224, [1, 3])
    output_b = slim.conv2d(output_b, 256, [3, 1])
    output_b = slim.conv2d(output_b, 2144, [1, 1], activation_fn=None)

    output = tf.add_n([output_a, output_b])
    output = tf.multiply(output, 0.1)

    return activation_fn(tf.add_n([output_res, output]))


def Average_Pooling(inputs):
    output = slim.avg_pool2d(inputs, [8, 8])
    return output


def Dropout(inputs, keep=0.8):
    output = slim.dropout(inputs, keep_prob=keep)
    return output





