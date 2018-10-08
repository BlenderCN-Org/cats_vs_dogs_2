from struct import *
import tensorflow as tf
def forward(inputs,n_classes):
    with slim.arg_scope([slim.conv2d],weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm):
        with tf.name_scope('Stem'):
            output=Stem(inputs)
        with tf.name_scope('5xInception_ResNet_A'):
            for i in range(5):
                output=Inception_ResNet_A(output)
        with tf.name_scope('Reduction_A'):
            output=Reduction_A(output)
        with tf.name_scope('10xReduction_B'):
            for i in range(10):
                output=Reduction_B(output)
        with tf.name_scope('Reduction_B'):
            output=Reduction_B(output)
        with tf.name_scope('5xInception_ResNet_C'):
            for i in range(5):
                output=Inception_ResNet_C(output)
        with tf.name_scope('AveragePooling'):
            output=Average_Pooling(output)
        with tf.name_scope('Dropout0.8'):
            output=Dropout(output,0.8)
            output=slim.flatten(output)
        with tf.name_scope('fc'):
            output=slim.fully_connected(output,n_classes)
        return output
