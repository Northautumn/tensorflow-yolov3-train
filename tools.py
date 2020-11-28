import numpy as np
import cv2
import glob
from lxml import etree
from config import Config


# 载入图像数据集，返回图像个数和图像路径集
def load_images(images_path=Config.IMAGES_PATH, image_type='jpg'):
    images_path = images_path if images_path[-1] == '/' else images_path + '/'
    images_set = glob.glob(images_path + '*.{}'.format(image_type))
    return len(images_set), images_set


# 载入xml文件，返回xml路径集
def load_xmls(xmls_path=Config.ANNOT_PATH):
    xmls_path = xmls_path if xmls_path[-1] == '/' else xmls_path + '/'
    xmls_set = glob.glob(xmls_path + '*.xml')
    return len(xmls_set), xmls_set


# 生成训练文本
def gen_train_txt(xmls, save_path=Config.TRAIN_TXT):
    dnames = dict()
    with open(Config.CLASSES_PATH, 'r') as data:
        for index, name in enumerate(data):
            dnames[name.strip('\n')] = index
    fd_writer = open(save_path, 'w+')
    for path in xmls:
        xml = open(path).read()
        sel = etree.HTML(xml)
        image_path = sel.xpath('//filename/text()')[0]
        names = sel.xpath('//name/text()')
        xmins = sel.xpath('//bndbox/xmin/text()')
        xmaxs = sel.xpath('//bndbox/xmax/text()')
        ymins = sel.xpath('//bndbox/ymin/text()')
        ymaxs = sel.xpath('//bndbox/ymax/text()')
        bndboxes = [image_path]
        for index, name in enumerate(names):
            x1 = xmins[index]
            y1 = ymins[index]
            x2 = xmaxs[index]
            y2 = ymaxs[index]
            name = dnames[name]
            bndboxes.append(','.join([x1, y1, x2, y2, str(name)]))
        txt = ' '.join(bndboxes)
        fd_writer.write(txt + '\n')
    fd_writer.close()


# 载入预训练文件
def load_pretrain_weights(model, weights_file='darknet53.conv.74'):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    j = 0
    for i in range(52):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
        if len(bn_weights) != 0:
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
        bn_layer = model.get_layer(bn_layer_name)
        j += 1
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        if len(conv_weights) != 0:
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
        if len(conv_weights) != 0:
            conv_layer.set_weights([conv_weights])
        if len(bn_weights) != 0:
            bn_layer.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


# 图像预处理
def image_preprocess(image, target_size, gt_boxes=None):
    # 416, 416
    ih, iw = target_size
    # 图像实际大小
    h, w, _ = image.shape
    # 获取缩放比例
    scale = min(iw/w, ih/h)
    # 获取缩放后的宽和高
    nw, nh = int(scale * w), int(scale * h)
    # 图像缩放
    image_resized = cv2.resize(image, (nw, nh))
    # 图像填充
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    # 图像归一化
    image_paded = image_paded / 255.
    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


if __name__ == '__main__':
    gen_train_txt(load_xmls())
