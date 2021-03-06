import tensorflow as tf
from utils import tf_melspectogram

FLAGS = tf.app.flags.FLAGS

l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.001)
bias_initializer = tf.contrib.layers.xavier_initializer()
kernel_initializer = tf.contrib.layers.xavier_initializer()


def activation_func(x):
    return tf.nn.leaky_relu(x, alpha=0.3)


def frequency_graph(x, is_training):
    with tf.variable_scope('Freq'):
        with tf.variable_scope('Layer1'):
            fconv1 = tf.layers.conv2d(
                x,
                filters=16,
                kernel_size=[10, 23],
                strides=(1, 1),
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l1_regularizer,
                bias_regularizer=l1_regularizer,
                name='conv'
            )
            fconv1 = activation_func(
                tf.layers.batch_normalization(fconv1, training=is_training, momentum=0.9))
            fpool1 = tf.layers.max_pooling2d(
                fconv1,
                pool_size=[2, 2],
                padding="valid",
                strides=(2, 2),
                name='pool'
            )

        with tf.variable_scope('Layer2'):
            fconv2 = tf.layers.conv2d(
                fpool1,
                filters=32,
                kernel_size=[5, 11],
                strides=(1, 1),
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l1_regularizer,
                bias_regularizer=l1_regularizer,
                name='conv'
            )
            fconv2 = activation_func(
                tf.layers.batch_normalization(fconv2, training=is_training, momentum=0.9))
            fpool2 = tf.layers.max_pooling2d(
                fconv2,
                pool_size=[2, 2],
                padding="valid",
                strides=(2, 2),
                name='pool'
            )

        with tf.variable_scope('Layer3'):
            fconv3 = tf.layers.conv2d(
                fpool2,
                filters=64,
                kernel_size=[3, 5],
                strides=(1, 1),
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l1_regularizer,
                bias_regularizer=l1_regularizer,
                name='conv'
            )
            fconv3 = activation_func(
                tf.layers.batch_normalization(fconv3, training=is_training, momentum=0.9))
            fpool3 = tf.layers.max_pooling2d(
                fconv3,
                pool_size=[2, 2],
                padding="valid",
                strides=(2, 2),
                name='pool'
            )

        with tf.variable_scope('Layer4'):
            fconv4 = tf.layers.conv2d(
                fpool3,
                filters=128,
                kernel_size=[2, 4],
                strides=(1, 1),
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l1_regularizer,
                bias_regularizer=l1_regularizer,
                name='conv'
            )
            fconv4 = activation_func(
                tf.layers.batch_normalization(fconv4, training=is_training, momentum=0.9))
            fpool4 = tf.layers.max_pooling2d(
                fconv4,
                pool_size=[1, 5],
                padding="valid",
                strides=(1, 5),
                name='pool'
            )

        fflat = tf.layers.flatten(fpool4)

    return fflat


def temporal_graph(x, is_training):
    with tf.variable_scope('Temp'):
        with tf.variable_scope('Layer1'):
            tconv1 = tf.layers.conv2d(
                x,
                filters=16,
                kernel_size=[21, 10],
                strides=(1, 1),
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l1_regularizer,
                bias_regularizer=l1_regularizer,
                name='conv'
            )
            tconv1 = activation_func(
                tf.layers.batch_normalization(tconv1, training=is_training, momentum=0.9))
            tpool1 = tf.layers.max_pooling2d(
                tconv1,
                pool_size=[2, 2],
                padding="valid",
                strides=(2, 2),
                name='pool'
            )

        with tf.variable_scope('Layer2'):
            tconv2 = tf.layers.conv2d(
                tpool1,
                filters=32,
                kernel_size=[10, 5],
                strides=(1, 1),
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l1_regularizer,
                bias_regularizer=l1_regularizer,
                name='conv'
            )
            tconv2 = activation_func(
                tf.layers.batch_normalization(tconv2, training=is_training, momentum=0.9))
            tpool2 = tf.layers.max_pooling2d(
                tconv2,
                pool_size=[2, 2],
                padding="valid",
                strides=(2, 2),
                name='pool'
            )

        with tf.variable_scope('Layer3'):
            tconv3 = tf.layers.conv2d(
                tpool2,
                filters=64,
                kernel_size=[5, 3],
                strides=(1, 1),
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l1_regularizer,
                bias_regularizer=l1_regularizer,
                name='conv'
            )
            tconv3 = activation_func(
                tf.layers.batch_normalization(tconv3, training=is_training, momentum=0.9))
            tpool3 = tf.layers.max_pooling2d(
                tconv3,
                pool_size=[2, 2],
                padding="valid",
                strides=(2, 2),
                name='pool'
            )

        with tf.variable_scope('Layer4'):
            tconv4 = tf.layers.conv2d(
                tpool3,
                filters=128,
                kernel_size=[4, 2],
                strides=(1, 1),
                padding="same",
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=l1_regularizer,
                bias_regularizer=l1_regularizer,
                name='conv'
            )
            tconv4 = activation_func(
                tf.layers.batch_normalization(tconv4, training=is_training, momentum=0.9))
            tpool4 = tf.layers.max_pooling2d(
                tconv4,
                pool_size=[5, 1],
                padding="valid",
                strides=(5, 1),
                name='pool'
            )

        tflat = tf.layers.flatten(tpool4)

    return tflat


def deep_nn(x, is_training):
    """deep_nn builds the graph for a deep net for classifying music genres.
  Args:
      x: an input tensor with the dimensions (N_examples, 6400), is the number of data points in the spectral space
  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the object images into one of 10 classes
        (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
    """
    x = tf_melspectogram(x)
    img_summary = tf.summary.image('Input_images', x)

    freq = frequency_graph(x, is_training)
    temp = temporal_graph(x, is_training)

    concat = tf.concat([freq, temp], 1)

    with tf.variable_scope('Fully_Connected'):
        drop = tf.layers.dropout(
            concat,
            rate=0.5,
            training=is_training
        )
        fc1 = tf.layers.dense(
            drop,
            units=200,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l1_regularizer,
            bias_regularizer=l1_regularizer,
            activation=activation_func,
            name='fc1'
        )
        fc1 = tf.layers.dropout(
            fc1,
            rate=0.25,
            training=is_training
        )
        fc2 = tf.layers.dense(
            fc1,
            units=FLAGS.num_classes,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=l1_regularizer,
            bias_regularizer=l1_regularizer,
            activation=None,
            name='fc2'
        )

    y = fc2

    return y, img_summary
