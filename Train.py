from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import time
import random
import numpy as np
import tensorflow as tf
import os

#import ModelBuilding
#import GenerateBatch    
    
TOTAL_EPOCHS = 75   
#TRAING_STEPS = 5000
TRAIN_SAMPLES = 28800
TEST_SAMPLES = 14400
CUT_TRAIN_BATCH_SIZE = 300
CUT_TEST_BATCH_SIZE = 300

BATCH_SIZE = 300
    
MODEL_SAVE_PATH = '/home/zmz/Pictures/Conv2d_0.2/model_corrctshffle_withFinalResult/'
MODEL_NAME = 'model20171221.ckpt'

def print_activations(t):
    print(t.op.name,'',t.get_shape().as_list())

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape = shape,dtype = tf.float32,
                                           stddev=0.1),name = 'weights')

def bias_variable(shape):
    return tf.Variable(tf.constant(0.0,shape = shape,dtype = tf.float32),
                       trainable = True,name = 'biases') 
    
def conv2d(x,W,strides):
    return tf.nn.conv2d(x,W,strides=strides,padding = 'SAME')
    
def max_pool(x,ksize,strides,name):
    return tf.nn.max_pool(x,ksize = ksize,strides = strides,padding = 'VALID',name = name )
    
def lrn(conv,name):
    return tf.nn.lrn(conv,4,bias = 1.0,alpha = 0.001/9,beta = 0.75,name = name)

def conv_inference(image_batch):
    #define Alexnet forward inference model,outputs conv results
    parameters = []
    image_batch_reshape = tf.reshape(image_batch,[300,160,160,1])
    with tf.name_scope('conv1') as scope:       
        kernel = weight_variable([11,11,1,64])
        conv = conv2d(image_batch_reshape,kernel,[1,4,4,1])
        biases = bias_variable([64])
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name = scope)
        print_activations(conv1)
        parameters += [kernel,biases]
    
    lrn1 = lrn(conv1,'lrn1')
    pool1 = max_pool(lrn1,[1,3,3,1],[1,2,2,1],'pool1')
    print_activations(pool1)
    
    with tf.name_scope('conv2')as scope:
        kernel = weight_variable([5,5,64,192])
        conv = conv2d(pool1,kernel,[1,1,1,1])
        biases = bias_variable([192])
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name = scope)
        print_activations(conv2)
        parameters += [kernel,biases]
    lrn2 = lrn(conv2,'lrn2')
    pool2 = max_pool(lrn2,[1,3,3,1],[1,2,2,1],'pool2')
    print_activations(pool2)
    
    with tf.name_scope('conv3') as scope:
        kernel = weight_variable([3,3,192,384])
        conv = conv2d(pool2,kernel,[1,1,1,1])
        biases = bias_variable([384])
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name = scope)
        print_activations(conv3)
        parameters += [kernel,biases]
    
    with tf.name_scope('conv4') as scope:
        kernel = weight_variable([3,3,384,256])
        conv = conv2d(conv3,kernel,[1,1,1,1])
        biases = bias_variable([256])
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name = scope)
        print_activations(conv4)
        parameters += [kernel,biases]
    
    with tf.name_scope('conv5') as scope:
        kernel = weight_variable([3,3,256,256])
        conv = conv2d(conv4,kernel,[1,1,1,1])
        biases = bias_variable([256])
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias,name = scope)
        print_activations(conv5)
        parameters += [kernel,biases]
    pool5 = max_pool(conv5,[1,3,3,1],[1,2,2,1],'pool5')
    print_activations(pool5)
    
    return pool5
        
def FC_inference(convsult):
    with tf.variable_scope('local6') as scope:
        reshape = tf.reshape(convsult,[300,-1])
        keep_prob = tf.placeholder(tf.float32)
        drop = tf.nn.dropout(reshape,keep_prob)
        dim = drop.get_shape()[1].value
        weights = weight_variable([dim,4096])
        biases = bias_variable([4096])
        fc6 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name = scope.name)
    
    with tf.variable_scope('local7')as scope:
        weights = weight_variable([4096,1024])
        biases = bias_variable([1024])
        fc7 = tf.nn.relu(tf.matmul(fc6,weights)+biases,name = scope.name)
    
    with tf.variable_scope('local8')as scope:
        weights = weight_variable([1024,3])
        biases = bias_variable([3])
        fc8 = tf.matmul(fc7,weights)+biases
    return fc8,keep_prob  
                

def next_batch(global_list,batch_size,Npy_Train_Images,Npy_Train_Labels,number):
    shuffle = global_list[number*batch_size:(number+1)*batch_size]
        
    #shuffle = random.sample(length,batch_size)
    Body_images =  Npy_Train_Images[shuffle[0],:]
    Body_labels = Npy_Train_Labels[shuffle[0],:]
    Body_images = Body_images[:,np.newaxis]
    Body_labels = Body_labels[:,np.newaxis]
    i = 1
    while i <= (batch_size-1):
        Paste_images = Npy_Train_Images[shuffle[i],:]
        Paste_labels = Npy_Train_Labels[shuffle[i],:]
        Paste_images = Paste_images[:,np.newaxis]
        Paste_labels = Paste_labels[:,np.newaxis]        
        Body_images = np.concatenate((Body_images,Paste_images),axis = 1)
        Body_labels = np.concatenate((Body_labels,Paste_labels),axis = 1)
        i = i+1
        
    return Body_images.transpose(),Body_labels.transpose()

def cut_batch(start,end,Images,Labels):
    batch_image = Images[start:end,:]
    batch_label = Labels[start:end,:]
    return batch_image,batch_label

def main(_):
    Npy_Train_Images = np.load('/home/zmz/Pictures/tmp/images4/TrainImage.npy')
    Npy_Train_Images = Npy_Train_Images.reshape([28800,160*160])
    Npy_Train_Labels = np.load('/home/zmz/Pictures/tmp/images4/TrainLabel.npy')
    Npy_Train_Labels = Npy_Train_Labels.reshape([28800,3])
    
    Npy_Test_Images = np.load('/home/zmz/Pictures/tmp/images4/TestImage.npy')
    Npy_Test_Images = Npy_Test_Images.reshape([14400,160*160])
    Npy_Test_Labels = np.load('/home/zmz/Pictures/tmp/images4/TestLabel.npy')
    Npy_Test_Labels = Npy_Test_Labels.reshape([14400,3])
    
    x = tf.placeholder(tf.float32,[None,160*160],name = 'x') 
    y_ = tf.placeholder(tf.float32,[None,3],name = 'y_')
    y_conv,keep_prob = FC_inference(conv_inference(x))
    #files = tf.train.match_filenames_once('/home/zmz/Pictures/Conv2d Alexnet/Data/data.tfrecords-*')
    
    #image_train,label_train,position_train = GenerateBatch.read_my_file_format(filename_queue)
    
    
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
        
    cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
    with tf.name_scope('accuracy'):
        my_correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
        correct_prediction = tf.cast(my_correct_prediction,tf.float32)
        
    accuracy = tf.reduce_mean(correct_prediction)
    

    saver = tf.train.Saver(max_to_keep = 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        #tf.initialize_all_variables().run()
        #coord = tf.train.Coordinator
        #threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        max_accuracy = 0.84
        length = list(range(TRAIN_SAMPLES))
        
        with open('/home/zmz/Pictures/Conv2d_0.2/acc.txt','w') as f:
            for epoch in range (TOTAL_EPOCHS):
                for number in range (int(TRAIN_SAMPLES/BATCH_SIZE)):
                    if number == 0:
                        random.shuffle(length)
                    current_training_step = epoch * 96 + number                
                    images_train,labels_train= next_batch(length,BATCH_SIZE,Npy_Train_Images,Npy_Train_Labels,number)
    #            if i % 50 == 0:
    #                train_accuracy,loss = sess.run([accuracy,cross_entropy],feed_dict={x:images_train,y_:labels_train,
    #                        keep_prob:1.0})
    #                print('step %d,training accuracy %g' % (i,train_accuracy),'loss on training batch is %g' %(loss))
                
                
                    if current_training_step % 50 == 0:
                        Total_accuracy_train = 0
                        Total_train_loss = 0
                        for k in range(int(TRAIN_SAMPLES/CUT_TRAIN_BATCH_SIZE)):
                            START = CUT_TRAIN_BATCH_SIZE * k
                            END = CUT_TRAIN_BATCH_SIZE * (k+1) 
                            image_batch,label_batch = cut_batch(START,END,Npy_Train_Images,Npy_Train_Labels)
                            train_accuracy,train_cross_entropy = sess.run([accuracy,cross_entropy],feed_dict = {x:image_batch,y_:label_batch,keep_prob:1.0})
                        
                            Total_accuracy_train += train_accuracy
                            Total_train_loss += train_cross_entropy
                        Total_accuracy_train /= 96
                        Total_train_loss /= 96
                        print('step %d,Total accuracy of training set is %g' % (current_training_step,Total_accuracy_train))
                        print('step %d,Total loss of training set is %g' % (current_training_step,Total_train_loss))
                    
                    if current_training_step % 50 == 0:
                        Total_accuracy_test = 0
                        Total_test_loss = 0
                        for k in range(int(TEST_SAMPLES/CUT_TEST_BATCH_SIZE)):
                            START = CUT_TEST_BATCH_SIZE * k
                            END = CUT_TEST_BATCH_SIZE * (k+1) 
                            image_batch,label_batch = cut_batch(START,END,Npy_Test_Images,Npy_Test_Labels)
                            test_accuracy,test_cross_entropy = sess.run([accuracy,cross_entropy],feed_dict = {x:image_batch,y_:label_batch,keep_prob:1.0})
                            Total_accuracy_test += test_accuracy
                            Total_test_loss += test_cross_entropy
                        Total_accuracy_test /= 48
                        Total_test_loss /= 48
                        print('step %d,Total accuracy of testing set is %g' % (current_training_step,Total_accuracy_test))
                        print('step %d,Total loss of testing set is %g'% (current_training_step,Total_test_loss))
                        
                        if Total_accuracy_test > max_accuracy:
                            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = (epoch*96+number))
                            max_accuracy = Total_accuracy_test
                            f.write('new high accuracy:'+str(max_accuracy)+'at step'+str(epoch*96+number)+'\n')
                            
                            #run evaluation
                            logitslist =np.empty([14400,3],dtype = int)  
                            for k in range(int(14400/300)):
                                START = 300 * k
                                END = 300 * (k+1) 
                                image_batch,label_batch = cut_batch(START,END,Npy_Test_Images,Npy_Test_Labels)
                                logits = sess.run(y_conv,feed_dict = {x:image_batch,y_:label_batch,keep_prob:1.0})
                                logitslist[START:END,:] = logits
                            
                            logitslist = np.argmax(logitslist,axis = 1)
                            New_Npy_Test_Labels = np.argmax(Npy_Test_Labels,axis = 1)
                            evaluate(logitslist,New_Npy_Test_Labels)
                    #一步训练
                    train_step.run(feed_dict={x:images_train,y_:labels_train, keep_prob:0.5})
#计算三类准确率的函数
def evaluate(logitslist,Npy_Test_Labels):
    print('In evaluate: shape logitslist:',logitslist.shape)
    print('In evaluate: shape Npy_Test_Labels',Npy_Test_Labels.shape)
    RATEDIC = {'NCRATE':0,'MCIRATE':0,'ADRATE':0}
    DISEASELIST = ['NCRATE','MCIRATE','ADRATE']
    classificationbox = [0,0,0]
    for i in range(3):
        for j in range(50):
            
            START = i*4800 + j*96
            END = i*4800 + (j+1)*96
            onepatient_logits = logitslist[START:END]
            count_logits = np.bincount(onepatient_logits)
            result_logits = np.argmax(count_logits)
            
            onepatient_labels = Npy_Test_Labels[START:END]
            count_labels = np.bincount(onepatient_labels)
            result_labels = np.argmax(count_labels)
            
            
            if result_logits == result_labels:
                classificationbox[result_logits]+=1
    for classes in range(3):
        classificationbox[classes] /= 50
    for i in range(3):
        RATEDIC[DISEASELIST[i]] = classificationbox[i]
    with open('/home/zmz/Pictures/Conv2d_0.2/FinalResults.txt','w') as f:
        print(RATEDIC,file = f)
        #print('\n',file = f)
            
#        print('Training is complete,now evaluate model on test images')
#        Total_accuracy_test = 0
#        for k in range(int(TEST_SAMPLES/CUT_TEST_BATCH_SIZE)):
#            START = CUT_TEST_BATCH_SIZE * k
#            END = CUT_TEST_BATCH_SIZE * (k+1) 
#            image_batch,label_batch = cut_batch(START,END,Npy_Test_Images,Npy_Test_Labels)
#            train_accuracy,my_correct_predictions = sess.run([accuracy,my_correct_prediction],feed_dict = {x:image_batch,y_:label_batch,keep_prob:1.0})
#            Total_accuracy_test += train_accuracy
#            printWrongClass(my_correct_predictions,k)
#        Total_accuracy_test /= 48
#        print('step %d,Total accuracy of testing set is %g' % (TRAING_STEPS,Total_accuracy_test))
        
        

                
                
#        print(sess.run(y_conv,feed_dict={x:imagebatch,y_:batch[1],
#                        keep_prob:1.0}),sess.run(y_,feed_dict={x:batch[0],y_:batch[1],
#                        keep_prob:1.0}))
#        print(accuracy.eval(feed_dict={x:batch[0],y_:batch[1],
#                        keep_prob:1.0}))
        #print('test accuracy %g' % accuracy.eval(feed_dict={x:Npy_Train_Images,
#            y_:Npy_Train_Labels,keep_prob:1.0}))
        #coord.request_stop()
        #coord.join(threads)
def printWrongClass(prediction,k):
    with open('/home/zmz/Pictures/Conv2d_0.2/console.txt','w') as f:
        sicklist = ['NC','MCI','AD']
        #patientnumber = list(range(50))
        #prediction = prediction.reshape(4)
        index = np.where(prediction == False)
       
        index = np.array(index)
        if  index.size == 0: 
            print(k*300+1,'到',(k+1)*300,'这三百个都对了')
            #print(k*300+1,'到',(k+1)*300,'这三百个都对了',file = f)
            f.write(str(k*300+1)+'到'+str((k+1)*300)+'这三百个都对了'+'\r\ n')
            
            
        else:
            
            index = index.reshape([-1])
            realindex = [(k*300 + i) for i in index]
            def h(x):
                return sicklist[(x//4800)]
            sick = list(map(h,realindex))
            def g(x):        
                x = x % 4800 # 0-4799
                y = x //96  #50个人中的第几个
                z = x % 96  #96张图的第几张
                return [y+1,z+1]
            details = list(map(g,realindex))
            
            for m in range(len(realindex)):
                print(sick[m],'类病人中的',details[m][0],'号的第',details[m][1],'张切片分类错误')
                #print(sick[m],'类病人中的',details[m][0],'号的第',details[m][1],'张切片分类错误',file = f )
                f.write(sick[m]+'类病人中的'+str(details[m][0])+'号的第'+str(details[m][1])+'张切片分类错误'+'\r\ n')
               

    
    
if __name__ == '__main__' :
    tf.app.run()
    
    
    
    
    
    


