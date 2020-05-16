import numpy as np
import tensorflow as tf
import os
from dataset import Dataset
from yolov3 import yolov3, decode, compute_loss
from config import Config
import tools
import shutil

trainset = Dataset()
logdir = './out/log'
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = Config.WARMUP_EPOCHS * steps_per_epoch
total_steps = Config.EPOCHS * steps_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3])
feature_maps = yolov3(input_tensor)
output_tensors = []
for i, fm in enumerate(feature_maps):
    pred_tensor = decode(fm, i)
    output_tensors.append(fm)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
tools.load_pretrain_weights(model)
# exit(0)

optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps.value(), optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps.value() < warmup_steps:
            lr = global_steps.value() / warmup_steps * Config.LR_INIT
        else:
            lr = Config.LR_END + 0.5 * (Config.LR_INIT - Config.LR_END) * (
                (1 + tf.cos((global_steps.value() - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


for epoch in range(Config.EPOCHS):
    for image_data, target in trainset:
        train_step(image_data, target)
    model.save_weights("./out/my_yolov3")
