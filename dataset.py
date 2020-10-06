import tensorflow as tf
import numpy as np
import os
import cv2
from config import Config
import tools


class Dataset:
    # 初始化配置参数
    def __init__(self):
        self.train_txt_path = Config.TRAIN_TXT
        self.train_input_size = Config.INPUT_SIZE
        self.batch_size = Config.BATCH_SIZE
        self.data_aug = Config.DATA_AUG
        self.anchor_per_scale = Config.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150
        self.strides = np.array(Config.STRIDES)
        self.classes = Config.read_classes_name()
        self.num_classes = Config.read_classes_num()
        # 读取anchors文件，返回3*3*2
        self.anchors = np.array(Config.read_anchors())

        self.num_samples, self.samples = tools.load_images()
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

        self.output_sizes = self.train_input_size // self.strides

        tools.gen_train_txt(tools.load_xmls())
        self.images_annots = self.load_train_txt()

        gpus = tf.config.list_physical_devices('GPU')
        self.gpu = None
        if gpus:
            self.gpu = gpus[0]
            tf.config.experimental.set_memory_growth(self.gpu, True)

    def load_train_txt(self):
        with open(self.train_txt_path, 'r') as reader:
            txt = reader.readlines()
            records = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(records)
        return records

    def __iter__(self):
        return self

    def __next__(self):
        if self.gpu:
            with tf.device('/gpu:0'):
                return self.gen_data()
        else:
            with tf.device('/cpu:0'):
                return self.gen_data()

    def gen_data(self):
        # batch_size*416*416*3
        batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)
        # batch_size*52*52*3*(5+80)
        batch_label_sbbox = np.zeros((self.batch_size, self.output_sizes[0], self.output_sizes[0],
                                      self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
        # batch_size*26*26*3*(5+80)
        batch_label_mbbox = np.zeros((self.batch_size, self.output_sizes[1], self.output_sizes[1],
                                      self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
        # batch_size*13*13*3*(5+80)
        batch_label_lbbox = np.zeros((self.batch_size, self.output_sizes[2], self.output_sizes[2],
                                      self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
        # batch_size * 3 * 4
        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
        # batch_size * 3 * 4
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
        # batch_size * 3 * 4
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

        num = 0
        # 处理批次
        if self.batch_count < self.num_batchs:
            # 处理每一批次大小
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples:
                    index -= self.num_samples
                image_annot = self.images_annots[index]
                # 解析每一个图像的annotation
                img, bboxes = self.parse_image_annot(image_annot)
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                batch_image[num, :, :, :] = img
                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
                num += 1
            self.batch_count += 1
            batch_smaller_target = batch_label_sbbox, batch_sbboxes
            batch_medium_target = batch_label_mbbox, batch_mbboxes
            batch_larger_target = batch_label_lbbox, batch_lbboxes

            return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
        else:
            self.batch_count = 0
            np.random.shuffle(self.images_annots)
            raise StopIteration

    def parse_image_annot(self, image_annot):
        list_image_annot = image_annot.split()
        # 图像路径
        image_path = Config.IMAGES_PATH + list_image_annot[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        # 读取图像
        image = cv2.imread(image_path)
        # 读取annotation，格式为[[xmin,ymin,xmax,ymax,c],...]
        bboxes = np.array([list(map(int, box.split(','))) for box in list_image_annot[1:]])

        if self.data_aug:
            pass

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 图像和bounding box预处理
        image, bboxes = tools.image_preprocess(np.copy(image),
                                               [self.train_input_size, self.train_input_size], np.copy(bboxes))
        # bboxes: 416尺寸里面的大小
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        # [[52,52,3,(5+80)],[26,26,3,(5+80)],[13,13,3,(5+80)]]
        label = [np.zeros((self.output_sizes[i], self.output_sizes[i],
                           self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        # [[150,4],[150,4],[150,4]]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        # [0,0,0]
        bbox_count = np.zeros((3,))
        # 处理图像中每一个bounding box
        for bbox in bboxes:
            # xmin,ymin,xmax,ymax
            bbox_coor = bbox[:4]
            # 类别
            bbox_class_ind = bbox[4]
            # [80]
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            # [x,y,w,h]
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # [[x52,y52,w52,h52],[x26,y26,w26,h26],[x13,y13,w13,h13]]
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                # [3,4]
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    @staticmethod
    def bbox_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def __len__(self):
        return self.num_batchs
