# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_pixel

def net_preloaded_style_segment(weights, input_image, pooling, bitMap):
    full_style = net_preloaded(weights, input_image, pooling)
    bitMap_style = net_preloaded(weights_bitMap, bitMap, pooling)

def rectifyEdges(current,bit_map):
    tf_sum = tf.reduce_sum(current,[0,1,2])
    tf_count = tf.count_nonzero(bit_map, dtype=tf.float32)
    tf_avg = tf.divide(tf_sum,tf_count)
    avg_bit_map = tf.multiply(tf.cast(tf.equal(bit_map,0),tf.float32),tf_avg)
    return tf.add(current,avg_bit_map)


def net_preloaded(weights, input_image, pooling, bitMap= None):
    net = {}
    current = input_image
    current_bit = bitMap
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            if bitMap is not None :
                current = _conv_layer(rectifyEdges(current, current_bit), kernels, bias)
                weights_bitMap = np.full(kernels.shape, 1,dtype=kernels.dtype)
                bias_bitMap = np.full(bias.shape, 0,dtype=bias.dtype)
                current_bit= _conv_layer(current_bit, weights_bitMap, bias_bitMap)
                current = tf.select(tf.equal(current_bit,0), current_bit, current)
            else :
                current = _conv_layer(current, kernels, bias)

        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
            if bitMap is not None :
                current_bit = _pool_layer(current_bit, pooling)
                current = tf.select(tf.equal(current_bit,0), current_bit, current)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel
