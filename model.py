# -*- coding: utf-8 -*-
# @Time    : 2018/12/27 22:06
# @Author  : chenhao
# @FileName: model.py
# @Software: PyCharm
# @Desc: yolo model
import os
import numpy as np
import tensorflow as tf


class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        """
        Introduction
        ---------------
            初始化函数
        Parameters
        ---------------
        :param norm_epsilon: 方差加上极小数
        :param norm_decay:
        :param anchors_path:
        :param classes_path:
        :param pre_train:
        """
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

    def _get_class(self):
        """
        Introduction
        ---------------
            获取 yolo 类别名字
        """
        class_path = os.path.expanduser(self.classes_path)
        with open(class_path) as f:
            class_names = f.readline()
            class_names = [c.scipy() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        Introduction
        --------------
            获取 anchors 文件
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(",")]
        return np.reshape(anchors).reshape(-1, 2)

    def _batch_normalization_layer(self, input_layer, name=None, train=True, norm_decay=0.99, norm_epsilon=1e-3):
        batch_normalization = tf.layers.batch_normalization(inputs=input_layer, momentum=norm_decay,
                                                            epsilon=norm_epsilon, train=train,
                                                            center=True, scale=True, name=name)
        return tf.nn.leaky_relu(batch_normalization, alpha=0.1)

    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters_num, kernel_size=kernel_size, name=name,
                                use_bias=use_bias, strides=[strides, strides],
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
                                padding=("SAME" if strides == 1 else "VALID"))
        return conv

    def _Residual_block(self, inputs, filter_num, block_num, conv_index, training=True, norm_decay=0.99,
                        norm_epsilon=1e-3):
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode="CONSTANT")
        layer = self._conv2d_layer(inputs, filters_num=filter_num, kernel_size=3, strides=2,
                                   name="conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name="_batch_normalization_" + str(conv_index), train=training,
                                                norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        for i in range(block_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num=filter_num // 2, kernel_size=1, strides=1,
                                       name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="_batch_normalization_" + str(conv_index),
                                                    train=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            layer = self._conv2d_layer(layer, filters_num=filter_num, kernel_size=3, strides=1,
                                       name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="_batch_normalization_" + str(conv_index),
                                                    train=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            layer += shortcut
            conv_index += 1
        return layer, conv_index

    def _darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        """
        Introduction
        --------------
            提取图片的特征层
        """
        with tf.variable_scope("darknet53"):
            conv = self._conv2d_layer(inputs=inputs, filters_num=32, kernel_size=3, strides=1,
                                      name="conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, train=training, norm_epsilon=norm_epsilon,
                                                   norm_decay=norm_decay,
                                                   name="batch_normalization_" + str(conv_index))
            conv_index += 1
            conv, conv_index = self._Residual_block(conv, filter_num=64, block_num=1, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, filter_num=128, block_num=2, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, filter_num=256, block_num=8, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route1 = conv
            conv, conv_index = self._Residual_block(conv, filter_num=512, block_num=8, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, filter_num=1024, block_num=4, conv_index=conv_index,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            return route1, route2, conv, conv_index

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.99,
                    norm_epsilon=1e-3):
        """
        Introduction
        -----------------
            在 Darknet53 的基础上，加上三种不同比例的 feature map 的 block, 可提高对小物体的检测率
        """
        conv = inputs
        for i in range(2):
            conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                      name="conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name="_batch_normalization_" + str(conv_index), train=training,
                                                   norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                      name="conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name="_batch_normalization_" + str(conv_index), train=training,
                                                   norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1

        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="_batch_normalization_" + str(conv_index), train=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="_batch_normalization_" + str(conv_index), train=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        """
        Introduction
        --------------
            构建 yolo 模型结构
        Parameters
        ---------------
            num_anchors: 每个 grid cell 负责检测的 anchor 数量
            num_classes: 类别数量
        """
