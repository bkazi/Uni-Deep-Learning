import tensorflow as tf
from utils import tf_melspectogram

FLAGS = tf.app.flags.FLAGS


def activation_func(x):
    return tf.nn.leaky_relu(x, alpha=0.3)


def shallow_nn(x, is_training):
    """shallow_nn builds the graph for a shallow net for classifying music genres.
  Args:
      x: an input tensor with the dimensions (N_examples, 6400), is the number of data points in the spectral space
  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the object images into one of 10 classes
        (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
    """
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001)
    x = tf_melspectogram(x)

    img_summary = tf.summary.image('Input_images', x)

    # Frequency
    with tf.variable_scope('Layer1_Freq'):
        fconv = tf.layers.conv2d(
            inputs=x,
            filters=16,
            kernel_size=[10, 23],
            strides=(1, 1),
            padding="same",
            activation=activation_func,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1),
            kernel_regularizer=l1_regularizer,
            bias_regularizer=l1_regularizer,
            name='fconv'
        )
        fpool = tf.layers.max_pooling2d(
            inputs=fconv,
            pool_size=[1, 20],
            padding="valid",
            strides=(1, 20),
            name='fpool'
        )
        fflat = tf.layers.flatten(fpool)

    with tf.variable_scope('Layer1_Temp'):
        tconv = tf.layers.conv2d(
            inputs=x,
            filters=16,
            kernel_size=[21, 20],
            strides=(1, 1),
            padding="same",
            activation=activation_func,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1),
            kernel_regularizer=l1_regularizer,
            bias_regularizer=l1_regularizer,
            name='tconv'
        )
        tpool = tf.layers.max_pooling2d(
            inputs=tconv,
            pool_size=[20, 1],
            padding="valid",
            strides=(20, 1),
            name='tpool'
        )
        tflat = tf.layers.flatten(tpool)

    concat = tf.concat([fflat, tflat], 1)

    with tf.variable_scope('Fully_Connected'):
        drop = tf.layers.dropout(
            inputs=concat,
            rate=0.1,
            training=is_training
        )
        fc1 = tf.layers.dense(
            inputs=drop,
            units=200,
            activation=activation_func,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1),
            kernel_regularizer=l1_regularizer,
            bias_regularizer=l1_regularizer,
            name='fc1'
        )
        fc2 = tf.layers.dense(
            inputs=fc1,
            units=FLAGS.num_classes,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
            bias_initializer=tf.initializers.constant(0.1),
            kernel_regularizer=l1_regularizer,
            bias_regularizer=l1_regularizer,
            activation=None,
            name='fc2'
        )

    y = fc2

    return y, img_summary
