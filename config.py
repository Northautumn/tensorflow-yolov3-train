import numpy as np


class Config:
    CLASSES_PATH = 'my.names'
    ANCHORS_PATH = 'baseline_anchors.txt'
    INPUT_SIZE = 416
    DATA_AUG = False
    BATCH_SIZE = 4
    WARMUP_EPOCHS = 2
    LR_INIT = 1e-3
    LR_END = 1e-6
    IMAGES_PATH = './data/images/'
    ANNOT_PATH = './data/annotations/'
    ANCHOR_PER_SCALE = 3
    STRIDES = [8, 16, 32]
    TRAIN_TXT = 'train.txt'
    EPOCHS = 4
    IOU_LOSS_THRESH = 0.5

    def __init__(self):
        pass

    # 读取类别文件
    @staticmethod
    def read_classes_name(classes_path=CLASSES_PATH):
        if 'CLASSES_NAME' in Config.__dict__:
            return Config.CLASSES_NAME
        else:
            names = dict()
            with open(classes_path, 'r') as reader:
                for index, name in enumerate(reader):
                    names[index] = name.strip('/n')
            Config.CLASSES_NAME = names
            Config.CLASSES_NUM = len(names)
            return Config.CLASSES_NAME

    @staticmethod
    def read_classes_num():
        if 'CLASSES_NUM' in Config.__dict__:
            return Config.CLASSES_NUM
        else:
            ns = len(Config.read_classes_name())
            Config.CLASSES_NUM = ns
            return Config.CLASSES_NUM

    # 读取anchors文件
    @staticmethod
    def read_anchors(anchors_path=ANCHORS_PATH):
        if 'ANCHORS' in Config.__dict__:
            return Config.ANCHORS
        else:
            with open(anchors_path) as reader:
                anchors = reader.readline()
            anchors = np.array(anchors.split(','), dtype=np.float32)
            Config.ANCHORS = anchors.reshape((3, 3, 2))
            return Config.ANCHORS


if __name__ == '__main__':
    print(Config.read_anchors())


