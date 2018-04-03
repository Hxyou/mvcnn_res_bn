#coding=utf-8
import tensorflow as tf
import re
import numpy as np
import globals as g_


# FLAGS = tf.app.flags.FLAGS
# # Basic model parameters.
# tf.app.flags.DEFINE_integer('batch_size', g_.BATCH_SIZE,
#                             """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_float('learning_rate', g_.INIT_LEARNING_RATE,
#                             """Initial learning rate.""")
#
#
# # Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
# LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# WEIGHT_DECAY_FACTOR = 0.004 / 5. # 3500 -> 2.8
#
# TOWER_NAME = 'tower'
# DEFAULT_PADDING = 'SAME'
#
#
# def _activation_summary(x):
#     """Helper to create summaries for activations.
#     Creates a summary that provides a histogram of activations.
#     Creates a summary that measure the sparsity of activations.
#     Args:
#       x: Tensor
#     Returns:
#       nothing
#     """
#     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#     # session. This helps the clarity of presentation on tensorboard.
#     tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
#     tf.summary.histogram(tensor_name + '/activations', x)
#     tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
#
# def _variable_on_cpu(name, shape, initializer):
#     """Helper to create a Variable stored on CPU memory.
#     Args:
#       name: name of the variable
#       shape: list of ints
#       initializer: initializer for Variable
#     Returns:
#       Variable Tensor
#     """
#     with tf.device('/cpu:0'):
#         var = tf.get_variable(name, shape, initializer=initializer)
#     return var
#
#
# def _variable_with_weight_decay(name, shape, wd):
#     """Helper to create an initialized Variable with weight decay.
#     Note that the Variable is initialized with a truncated normal distribution.
#     A weight decay is added only if one is specified.
#     Args:
#       name: name of the variable
#       shape: list of ints
#       wd: add L2Loss weight decay multiplied by this float. If None, weight
#           decay is not added for this Variable.
#     Returns:
#       Variable Tensor
#     """
#     var = _variable_on_cpu(name, shape,
#                            initializer=tf.contrib.layers.xavier_initializer())
#     if wd:
#         weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#         tf.add_to_collection('losses', weight_decay)
#     return var
#
#
# def _conv(name, in_ ,ksize, strides=[1,1,1,1], padding=DEFAULT_PADDING, group=1, reuse=False):
#
#     n_kern = ksize[3]
#     convolve = lambda i, k: tf.nn.conv2d(i, k, strides, padding=padding)
#
#     with tf.variable_scope(name, reuse=reuse) as scope:
#         if group == 1:
#             kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
#             conv = convolve(in_, kernel)
#         else:
#             ksize[2] /= group
#             kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
#             input_groups = tf.split(in_, group, 3)
#             kernel_groups = tf.split(kernel, group, 3)
#             output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
#             # Concatenate the groups
#             conv = tf.concat(output_groups, 3)
#
#         biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
#         conv = tf.nn.bias_add(conv, biases)
#         conv = tf.nn.relu(conv, name=scope.name)
#         _activation_summary(conv)
#
#     #print (name, conv.get_shape().as_list())
#     return conv
#
# def _maxpool(name, in_, ksize, strides, padding=DEFAULT_PADDING):
#     pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
#                           padding=padding, name=name)
#
#     #print (name, pool.get_shape().as_list())
#     return pool
#
# def _fc(name, in_, outsize, dropout=1.0, reuse=False):
#     with tf.variable_scope(name, reuse=reuse) as scope:
#         # Move everything into depth so we can perform a single matrix multiply.
#
#         insize = in_.get_shape().as_list()[-1]
#         weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
#         biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
#         fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
#         fc = tf.nn.dropout(fc, dropout)
#
#         _activation_summary(fc)
#
#
#
#     #print (name, fc.get_shape().as_list())
#     return fc
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', g_.BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', g_.INIT_LEARNING_RATE,
                            """Initial learning rate.""")

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
WEIGHT_DECAY_FACTOR = 0.004 / 5. # 3500 -> 2.8

TOWER_NAME = 'tower'
DEFAULT_PADDING = 'SAME'

# def batch_norm(inputs, training, data_format):
#   """Performs a batch normalization using a standard set of parameters."""
#   # We set fused=True for a significant performance boost. See
#   # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
#   return tf.layers.batch_normalization(
#       inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
#       momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
#       scale=True, training=training, fused=True)

# def batch_norm(inputs, training, data_format):
#   """Performs a batch normalization using a standard set of parameters."""
#   # We set fused=True for a significant performance boost. See
#   # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
#   return inputs



def batch_norm(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed



def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs



def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format, bn_decay):
  """A single block for ResNet v2, without a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  # inputs = batch_norm(inputs, training, data_format)
  inputs = batch_norm(inputs, training, scope='bn_1', moments_dims=[0, 1, 2], bn_decay=bn_decay)

  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  # inputs = batch_norm(inputs, training, data_format)
  inputs = batch_norm(inputs, training, scope='bn_2', moments_dims=[0, 1, 2], bn_decay=bn_decay)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  # inputs = batch_norm(inputs, training, data_format)
  inputs = batch_norm(inputs, training, scope='bn_3', moments_dims=[0, 1, 2], bn_decay=bn_decay)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut



def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, bn_decay=None):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format, bn_decay)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format, bn_decay)

  return tf.identity(inputs, name)




def inference_multiview(images, n_classes=40, is_training=False, bn_decay=None):
    """
    :param images: (N, V, W, H, C)
    :param is_training: is_training
    :param bn_decay: bn_decay
    :return: fc logits
    """

    batch_size = images.get_shape().as_list()[0]
    n_views = images.get_shape().as_list()[1]
    weight = images.get_shape().as_list()[2]
    height = images.get_shape().as_list()[3]
    dims = images.get_shape().as_list()[4]
    print (images.get_shape().as_list())

    bottleneck = True
    num_filters = 64
    kernel_size = 7
    conv_stride = 2
    first_pool_size = 3
    first_pool_stride = 2
    second_pool_size = 7
    second_pool_stride = 1
    block_sizes = [3, 4, 6, 3]
    block_strides = [1, 2, 2, 2]
    final_size = 2048
    version = 2
    data_format = 'channels_last'
    num_classes = n_classes


    # Get images (N*V, W, H, C)
    images = tf.reshape(images, [-1, weight, height, dims])
    print (images.get_shape().as_list())

    # reuse = False


    inputs = conv2d_fixed_padding(
        inputs=images, filters=num_filters, kernel_size=kernel_size,
        strides=conv_stride, data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    print (inputs.get_shape().as_list())


    if first_pool_size:
      inputs = tf.layers.max_pooling2d(
          inputs=inputs, pool_size=first_pool_size,
          strides=first_pool_stride, padding='SAME',
          data_format=data_format)
      inputs = tf.identity(inputs, 'initial_max_pool')

    print (inputs.get_shape().as_list())


    for i, num_blocks in enumerate(block_sizes):
      num_filters1 = num_filters * (2**i)
      inputs = block_layer(
          inputs=inputs, filters=num_filters1, bottleneck=bottleneck,
          block_fn=_bottleneck_block_v2, blocks=num_blocks,
          strides=block_strides[i], training=is_training,
          name='block_layer{}'.format(i + 1), data_format=data_format, bn_decay=bn_decay)
      print (inputs.get_shape().as_list())

    # inputs = batch_norm(inputs, is_training, data_format)
    inputs = batch_norm(inputs, is_training, scope='bn_0', moments_dims=[0,1,2], bn_decay=bn_decay)
    inputs = tf.nn.relu(inputs)

    # The current top layer has shape
    # `batch_size x pool_size x pool_size x final_size`.
    # ResNet does an Average Pooling layer over pool_size,
    # but that is the same as doing a reduce_mean. We do a reduce_mean
    # here because it performs better than AveragePooling2D.
    axes = [2, 3] if data_format == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keep_dims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')
    print (inputs.get_shape().as_list())


    hwc = np.prod(inputs.get_shape().as_list()[1:])
    flatten = tf.reshape(inputs, [-1, hwc])
    view_pooling = tf.reshape(flatten, [-1, n_views, hwc])
    view_pooling = tf.reduce_max(view_pooling, axis=1)
    print (view_pooling.get_shape().as_list())


    inputs = tf.reshape(view_pooling, [-1, final_size])
    print (inputs.get_shape().as_list())

    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')

    return inputs


# def inference_multiview(views, n_classes, keep_prob):
#     """
#     views: N x V x W x H x C tensor
#     """
#     n_views = views.get_shape().as_list()[1]
#
#     # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
#     views = tf.transpose(views, perm=[1, 0, 2, 3, 4])
#
#     view_pool = []
#     for i in range(n_views):
#         # set reuse True for i > 0, for weight-sharing
#         reuse = (i != 0)
#         view = tf.gather(views, i) # NxWxHxC
#
#         conv1 = _conv('conv1', view, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
#         lrn1 = None
#         pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#         conv2 = _conv('conv2', pool1, [5, 5, 96, 256], group=2, reuse=reuse)
#         lrn2 = None
#         pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#         conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
#         conv4 = _conv('conv4', conv3, [3, 3, 384, 384], group=2, reuse=reuse)
#         conv5 = _conv('conv5', conv4, [3, 3, 384, 256], group=2, reuse=reuse)
#
#         pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#         dim = np.prod(pool5.get_shape().as_list()[1:])
#         reshape = tf.reshape(pool5, [-1, dim])
#
#         view_pool.append(reshape)
#
#
#     pool5_vp = _view_pool(view_pool, 'pool5_vp')
#     #print ('pool5_vp', pool5_vp.get_shape().as_list())
#
#
#     fc6 = _fc('fc6', pool5_vp, 4096, dropout=keep_prob)
#     fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob)
#     fc8 = _fc('fc8', fc7, n_classes)
#
#     return fc8
    

def load_alexnet_to_mvcnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath, encoding='latin1')
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = l
        _load_param(sess, name, data_dict[l])
    

def _load_param(sess, name, layer_data):
    w, b = layer_data

    with tf.variable_scope(name, reuse=True):
        for subkey, data in zip(('weights', 'biases'), (w, b)):
            #print ('loading ', name, subkey)

            try:
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))
            except ValueError as e: 
                print ('varirable loading failed:', subkey, '(%s)' % str(e))


# def _view_pool(view_features, name):
#     vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
#     for v in view_features[1:]:
#         v = tf.expand_dims(v, 0)
#         vp = tf.concat([vp, v], 0)
#     #print ('vp before reducing:', vp.get_shape().as_list())
#     vp = tf.reduce_max(vp, [0], name=name)
#     return vp


def loss(fc8, labels):
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc8)
    l = tf.reduce_mean(l)

    return l

    # tf.add_to_collection('losses', l)
    #
    # return tf.add_n(tf.get_collection('losses'), name='total_loss')


def classify(fc8):
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    #print ('losses:', losses)
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
    

def train(total_loss, global_step, data_size):
    num_batches_per_epoch = data_size / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    
    # loss_averages_op = _add_loss_summaries(total_loss)

    # with tf.control_dependencies([loss_averages_op]):
    #     opt = tf.train.AdamOptimizer(lr)
        # grads = opt.compute_gradients(total_loss)

    opt = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # train_op = tf.group(opt.minimize(total_loss, global_step), update_ops)

    with tf.control_dependencies(update_ops):
      train_op = opt.minimize(total_loss, global_step=global_step)

    # train_op = opt.minimize(total_loss, global_step=global_step)
    # grads = opt.compute_gradients(total_loss)
    # #
    # #
    # # # apply gradients
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # #
    # #
    # # for var in tf.trainable_variables():
    # #     tf.summary.histogram(var.op.name, var)
    # #
    # # for grad,var in grads:
    # #     if grad is not None:
    # #         tf.summary.histogram(var.op.name + '/gradients', grad)
    # #
    # # # variable_averages = tf.train.ExponentialMovingAverage(
    # # #         MOVING_AVERAGE_DECAY, global_step)
    # # #
    # # # variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # #
    # # # with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
    # with tf.control_dependencies([apply_gradient_op]):
    #     train_op = tf.no_op(name='train')

    return train_op
