import tensorflow as tf
import numpy as np
import vgg19_model
import os
import random

BATCH_SIZE = 300
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.98
REGULARIZATION_RATE = 0.01  #lambda
KEEP_PROB = 0.5
TRAINING_STEPS = 2000

MODEL_SAVE_PATH="/home/yxq/vgg19new/vgg19model"
MODEL_NAME="level_1000_lambda1e-2.ckpt"

def load_train_img():
    train_img = np.load("/home/yxq/trainConvOut96.npy")
    train_label = np.zeros([96*300, 3])
    train_label[0:96*100, 2] = 1
    train_label[96*100:96*200, 1] = 1
    train_label[96*200:96*300, 0] = 1
    
    return train_img, train_label

def train():
    
    global_step = tf.Variable(0, trainable=False)

    x_holder = tf.placeholder(tf.float32, [BATCH_SIZE, 5, 5, 512], name='x-input')
    y_holder = tf.placeholder(tf.float32, [BATCH_SIZE, 3], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y_train = vgg19_model.fcResult(x_holder, regularizer, KEEP_PROB)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_train, labels=y_holder)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss', loss)  #用于tensorboard

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        20*300*96//BATCH_SIZE,
        LEARNING_RATE_DECAY,
    )

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver() 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_arr, train_label = load_train_img()
        print('img loaded!')
        merged = tf.summary.merge_all()
        # 选定可视化存储目录
        writer = tf.summary.FileWriter("/home/yxq/tensorboard", sess.graph)
        for p in range(TRAINING_STEPS):
            for q in range(300*96//BATCH_SIZE):
                _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x_holder: train_arr[q*BATCH_SIZE:(q+1)*BATCH_SIZE,:,:,:], y_holder: train_label[q*BATCH_SIZE:(q+1)*BATCH_SIZE,:]})
            print("Round %d, after %d training step(s), loss on training batch is %g." % (p+1, step, loss_value))
            merged_result = sess.run(merged)  # merged也是需要run的
            writer.add_summary(merged_result, p)
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

if __name__ == '__main__':
    train()
