#前50个是测试集，后100个是训练集

import numpy as np
import os
import dicom
data_path =['/home/zmz/Pictures/%s' %i for i in ['NC','MCI','AD']]
label_list = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype = np.int16)

train_img = np.zeros([3,100,96,160*160])
train_label = np.zeros([3,100,96,3])
test_img = np.zeros([3,50,96,160*160])
test_label = np.zeros([3,50,96,3])

#SLICE_PICK = 48

for p in range(len(data_path)):
        Patientlist = os.listdir(data_path[p])
        
        for q in range(len(Patientlist)):
            Dicompath = os.path.join(data_path[p],Patientlist[q])
            Dicomlist = os.listdir(Dicompath)
            
            for k in range(len(Dicomlist)):
               
                Picturepath = os.path.join(Dicompath,Dicomlist[k])
                img = dicom.read_file(Picturepath)
                imgpixel = img.pixel_array
                imgpixel = imgpixel.astype(np.int32)
                #print(type(img.pixel_array),img.pixel_array.shape,img.pixel_array.dtype)
    #                pil_img = Image.fromarray(imgpixel)
    #                imgpixel = np.array(pil_img.resize((80,80)))
                mean = np.mean(imgpixel)
                #minimum = np.amin(imgpixel)
                #maximum = np.amax(imgpixel)
                #print(q,minimum,maximum)
                std = np.std(imgpixel)
                std += 1e-5
                if abs(std - 0) < 1e-7:
                    print('std is near 0')
                imgpixel = imgpixel-mean
                imgpixel = imgpixel / std
                #print(np.amin(imgpixel),np.amax(imgpixel)) 
                #imgpixel = imgpixel.astype(np.float32)
                imgpixel = imgpixel.reshape(160*160)
                #imgpixel = imgpixel.astype(np.float32,casting = 'same_kind')
                #print (np.unique(imgpixel))
                if q<50:
                    test_img[p,q,k,:]=imgpixel
                    test_label[p,q,k:]=label_list[p]
                else:
                    train_img[p,(q-50),k,:] = imgpixel
                    train_label[p,(q-50),k,:]=label_list[p]                  

#train_img = train_img.astype(np.float32)
#train_label = train_label.astype(np.float32)

train_img = train_img.reshape([300*96,160*160])
train_label = train_label.reshape([300*96,3])
test_img = test_img.reshape([150*96,160*160])
test_label = test_label.reshape([150*96,3])

train_label =train_label.astype(np.int32)
test_label = test_label.astype(np.int32)
train_img = train_img.astype(np.float32,casting = 'same_kind')
test_img = test_img.astype(np.float32,casting = 'same_kind')
print ("train images信息：",train_img.dtype,train_img.shape)
print ("train labels信息：",train_label.dtype,train_label.shape)
print ("test images信息：",test_img.dtype,test_img.shape)
print ("test labels信息：",test_label.dtype,test_label.shape)
#train_img = train_img.astype(np.float32)
#train_label = train_label.astype(np.float32)
#print (train_img.dtype)
#print (train_label.dtype)
#print (np.unique(train_img)) 
#print (np.unique(train_label))              
np.save('/home/zmz/Pictures/tmp/images4/TrainImage',train_img)
np.save('/home/zmz/Pictures/tmp/images4/TrainLabel',train_label)
np.save('/home/zmz/Pictures/tmp/images4/TestImage',test_img)
np.save('/home/zmz/Pictures/tmp/images4/TestLabel',test_label)
            
 

