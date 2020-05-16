# tensorflow-yolov3-train

#### 测试环境

tensorflow == 2.1.0
numpy == 1.16.4
lxml == 4.5.0
opencv-python == 4.2.0



#### 注:

代码大部分参考了`YunYang1994/TensorFlow2.0-Examples/4-Object_Detection/YOLOV3`中的代码， 本人只是在别人基础上稍微调整了一下符合自己的需求。

有很多人直接利用自己的图像数据集来训练YOLOV3的网络，结果很不理想或者检测不到物体又或者检测的物体置信度很低，这里建议载入预训练权重再进行训练，结果会有改善。项目默认需要载入预训练权重才能正常训练。



#### 步骤

1. 首先下载预训练权重(`darknet53.conv.74`)，放到项目根目录下(注：建议使用预训练权重)。
   链接:https://pan.baidu.com/s/1G7-EG-XNswF9ZcGmxTq9-Q  密码:ltg6
2. 图像放到`data/images`下，xml文件放到`data/annotations`下。(注：暂时没有考虑到验证和测试相关，本项目只用于自己测试。)
3. 运行`train.py`即可，会在项目根目录下生成`out`文件夹，里面存放结果。



#### 参考

- https://github.com/YunYang1994/TensorFlow2.0-Examples