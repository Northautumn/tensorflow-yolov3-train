from easydict import EasyDict

Config = EasyDict()
Config.CLASSES = 'my.names'
Config.ANCHORS = 'baseline_anchors.txt'
Config.INPUT_SIZE = 416
Config.DATA_AUG = False
Config.BATCH_SIZE = 4
Config.WARMUP_EPOCHS = 2
Config.LR_INIT = 1e-3
Config.LR_END = 1e-6
Config.IMAGES_PATH = './data/images/'
Config.ANNOT_PATH = './data/annotations/'
Config.ANCHOR_PER_SCALE = 3
Config.STRIDES = [8, 16, 32]
Config.TRAIN_TXT = 'train.txt'
Config.EPOCHS = 4
Config.IOU_LOSS_THRESH = 0.5

