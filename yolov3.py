import tensorflow as tf
import numpy as np
from common import convolutional, upsample
from darknet import darknet53
from tools import read_classes_names, read_anchors
from config import Config

NUM_CLASS = len(read_classes_names())
STRIDES = np.array(Config.STRIDES)
ANCHORS = read_anchors()


def yolov3(inputs):
    route1, route2, route3 = darknet53(inputs)

    conv = convolutional(route3, 512, (1, 1))
    conv = convolutional(conv, 1024, (3, 3))
    conv = convolutional(conv, 512, (1, 1))
    conv = convolutional(conv, 1024, (3, 3))
    conv = convolutional(conv, 512, (1, 1))

    conv_lobj_branch = convolutional(conv, 1024, (3, 3))
    conv_lbbox = convolutional(conv_lobj_branch, 3*(NUM_CLASS + 5), (1, 1), activate=False, bn=False)

    conv = convolutional(conv, 256, (1, 1))
    conv = upsample(conv)

    conv = tf.concat([conv, route2], axis=-1)

    conv = convolutional(conv, 256, (1, 1))
    conv = convolutional(conv, 512, (3, 3))
    conv = convolutional(conv, 256, (1, 1))
    conv = convolutional(conv, 512, (3, 3))
    conv = convolutional(conv, 256, (1, 1))

    conv_mobj_branch = convolutional(conv, 512, (3, 3))
    conv_mbbox = convolutional(conv_mobj_branch, 3*(NUM_CLASS + 5), (1, 1), activate=False, bn=False)

    conv = convolutional(conv, 128, (1, 1))
    conv = upsample(conv)

    conv = tf.concat([conv, route1], axis=-1)

    conv = convolutional(conv, 128, (1, 1))
    conv = convolutional(conv, 256, (3, 3))
    conv = convolutional(conv, 128, (1, 1))
    conv = convolutional(conv, 256, (3, 3))
    conv = convolutional(conv, 128, (1, 1))

    conv_sobj_branch = convolutional(conv, 256, (3, 3))
    conv_sbbox = convolutional(conv_sobj_branch, 3*(NUM_CLASS + 5), (1, 1), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


# boxes = [center_x, center_y, w, h]
def bbox_iou(boxes1, boxes2):
    # 计算两个框的大小
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 转换成两个顶点模式
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # 计算左上，右下坐标
    left_top = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_bottom = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    # 取出交叉区域
    inter_section = tf.maximum(right_bottom - left_top, 0.0)
    # 计算交叉面积
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    # 计算联合面积
    union_area = boxes1_area + boxes2_area - inter_area
    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    return giou


def decode(conv_output, i=0):
    # conv_output = batch_size * output_size * output_size * (3 * (num_class + 5))
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_confidence = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_confidence)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def compute_loss(pred, conv, label, bboxes, i=0):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # 模型输出的置信值与分类
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    # 模型输出处理后预测框的位置
    pred_xywh = pred[:, :, :, :, 0:4]
    # 模型输出处理后预测框的置信值
    pred_conf = pred[:, :, :, :, 4:5]

    # 标签图片的标注框位置
    label_xywh = label[:, :, :, :, 0:4]
    # 标签图片的置信值，有目标的为1 没有目标为0
    respond_bbox = label[:, :, :, :, 4:5]
    # 标签图片的分类
    label_prob = label[:, :, :, :, 5:]

    # 框回归损失
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    # 置信度损失
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < Config.IOU_LOSS_THRESH, tf.float32)
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    # 分类损失
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
