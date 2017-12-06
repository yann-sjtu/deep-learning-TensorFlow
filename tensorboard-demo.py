import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


layer_name = 'demo-layer'
with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
        w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros([1]))
    y = w * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.summary.histogram(layer_name + "/weights", w)
    tf.summary.histogram(layer_name + "/bias", b)
    tf.summary.scalar('loss', loss)  # 命名和赋值
    merged = tf.summary.merge_all()
    # 选定可视化存储目录
    writer = tf.summary.FileWriter("/Users/yann/Desktop/python/tensorboard-demo", sess.graph)
    for step in range(100):
        sess.run(train_op)
        result = sess.run(merged)  # merged也是需要run的
        writer.add_summary(result, step)