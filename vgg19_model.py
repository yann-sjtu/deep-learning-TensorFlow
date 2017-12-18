import tensorflow as tf

CONV1_DEEP = 64
CONV2_DEEP = 128
CONV3_DEEP = 256
CONV4_DEEP = 512
CONV5_DEEP = 512  #卷积后的图像深度
CONV6_DEEP = 128

FC1_SIZE = 300
FC2_SIZE = 128

TAIL1_SIZE = 64
TAIL2_SIZE = 16
TAIL3_SIZE = 3
#OUTPUT_NODE = 1000   #全连接层的输出节点数
KEEP_PROB = 1.0
CLASS_NUM = 3
CHANNEL_NUM = 3

def convLayer(x, kHeight, kWidth, strideX, strideY,  channelNum,
              featureNum, name, padding = "SAME"):  
    """convlutional"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channelNum, featureNum])  
        b = tf.get_variable("b", shape = [featureNum])  
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)  
        out = tf.nn.bias_add(featureMap, b)  
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)
    
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):  
    """max-pooling"""  
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],  
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)  

def dropout(x, keepPro, name = None):  
    """dropout"""  
    return tf.nn.dropout(x, keepPro, name)  
 
def fcLayer(x, inputD, outputD, reluFlag, regularizer, name):
    """fully-connect"""  
    with tf.variable_scope(name) as scope:  
        w = tf.get_variable("w", shape = [inputD, outputD], initializer = tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', regularizer(w))
        b = tf.get_variable("b", [outputD], initializer = tf.constant_initializer(0.1))  
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:  
            return tf.nn.relu(out), w, b
        else:  
            return out, w, b
        
  
def convResult(x_input, ksize, stride, pool_size, pool_stride, keep_prob = KEEP_PROB, class_num = CLASS_NUM):
    """define vgg19 model"""
    #x_input是n张rgb图组成的batch
    conv1_1 = convLayer(x_input, ksize, ksize, stride, stride, CHANNEL_NUM, CONV1_DEEP, "conv1_1" )  
    conv1_2 = convLayer(conv1_1, ksize, ksize, stride, stride, CONV1_DEEP, CONV1_DEEP, "conv1_2")  
    pool1 = maxPoolLayer(conv1_2, pool_size, pool_size, pool_stride, pool_stride, "pool1")  
  
    conv2_1 = convLayer(pool1, ksize, ksize, stride, stride, CONV1_DEEP, CONV2_DEEP, "conv2_1")  
    conv2_2 = convLayer(conv2_1, ksize, ksize, stride, stride, CONV2_DEEP, CONV2_DEEP, "conv2_2")  
    pool2 = maxPoolLayer(conv2_2, pool_size, pool_size, pool_stride, pool_stride, "pool2")  
  
    conv3_1 = convLayer(pool2, ksize, ksize, stride, stride, CONV2_DEEP, CONV3_DEEP, "conv3_1")  
    conv3_2 = convLayer(conv3_1, ksize, ksize, stride, stride, CONV3_DEEP, CONV3_DEEP, "conv3_2")  
    conv3_3 = convLayer(conv3_2, ksize, ksize, stride, stride, CONV3_DEEP, CONV3_DEEP, "conv3_3")  
    conv3_4 = convLayer(conv3_3, ksize, ksize, stride, stride, CONV3_DEEP, CONV3_DEEP, "conv3_4")
    pool3 = maxPoolLayer(conv3_4, pool_size, pool_size, pool_stride, pool_stride, "pool3")  
  
    conv4_1 = convLayer(pool3, ksize, ksize, stride, stride, CONV3_DEEP, CONV4_DEEP, "conv4_1")  
    conv4_2 = convLayer(conv4_1, ksize, ksize, stride, stride, CONV4_DEEP, CONV4_DEEP, "conv4_2")  
    conv4_3 = convLayer(conv4_2, ksize, ksize, stride, stride, CONV4_DEEP, CONV4_DEEP, "conv4_3")  
    conv4_4 = convLayer(conv4_3, ksize, ksize, stride, stride, CONV4_DEEP, CONV4_DEEP, "conv4_4")  
    pool4 = maxPoolLayer(conv4_4, pool_size, pool_size, pool_stride, pool_stride, "pool4")  
  
    conv5_1 = convLayer(pool4, ksize, ksize, stride, stride, CONV4_DEEP, CONV5_DEEP, "conv5_1")  
    conv5_2 = convLayer(conv5_1, ksize, ksize, stride, stride, CONV5_DEEP, CONV5_DEEP, "conv5_2")  
    conv5_3 = convLayer(conv5_2, ksize, ksize, stride, stride, CONV5_DEEP, CONV5_DEEP, "conv5_3")  
    conv5_4 = convLayer(conv5_3, ksize, ksize, stride, stride, CONV5_DEEP, CONV5_DEEP, "conv5_4")  
    pool5 = maxPoolLayer(conv5_4, pool_size, pool_size, pool_stride, pool_stride, "pool5")  

    return pool5
    

def fcResult(conv_out, regularizer, keep_prob = KEEP_PROB, class_num = CLASS_NUM):

    conv6 = conv_out

    fc_shape = conv6.get_shape().as_list()
    nodes = fc_shape[1] * fc_shape[2] * fc_shape[3]
    fc_in = tf.reshape(conv6, [fc_shape[0], nodes])

    fc6, w6, b6 = fcLayer(fc_in, nodes, FC1_SIZE, True, regularizer, "fc6")

    fc7, w7, b7 = fcLayer(fc6, FC1_SIZE, FC2_SIZE, True, regularizer, "fc7")
    dropout2 = dropout(fc7, keep_prob)  
  
    fc8, w8, b8 = fcLayer(dropout2, FC2_SIZE, class_num, False, regularizer, "fc8")
    params=[w6, b6, w7, b7, w8, b8]

    return fc8, params

def tailResult(tail_input, regularizer, keep_prob=KEEP_PROB):
    
    fc1 = fcLayer(tail_input, 288, TAIL1_SIZE, True, regularizer, 'fc1')

    fc2 = fcLayer(fc1, TAIL1_SIZE, TAIL2_SIZE, True, regularizer, 'fc2')

    dropout2 = dropout(fc2, keep_prob)

    fc3 = fcLayer(dropout2, TAIL2_SIZE, TAIL3_SIZE, False, regularizer, 'fc3')

    return fc3


