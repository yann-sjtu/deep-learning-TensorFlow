import numpy as np
import os
import scipy.io as sio

ad_path = '/home/yxq/MRI_IMAGE/AD_MRI'
mci_path = '/home/yxq/MRI_IMAGE/MCI_MRI'
nc_path = '/home/yxq/MRI_IMAGE/NC_MRI'

def load_img():
    img_arr = np.zeros([170*3, 256, 256, 166])
    n = 0
    print('start')
    for matfile in os.listdir(ad_path)[:170]:
        img_arr[n, :, :, :] = sio.loadmat(matfile)['Y']
        n += 1
    for matfile in os.listdir(mci_path)[:170]:
        img_arr[n, :, :, :] = sio.loadmat(matfile)['Y']
        n += 1
    for matfile in os.listdir(nc_path)[:170]:
        img_arr[n, :, :, :] = sio.loadmat(matfile)['Y']
        n += 1
    print("load over")
    trainImageArray = np.zeros(120*3, 256, 256, 166)
    testImageArray = np.zeros(50*3, 256, 256, 166)

    trainImageArray[:120, :, :, :] = img_arr[:120, :, :, :]
    trainImageArray[120:240, :, :, :] = img_arr[170:290, :, :, :]
    trainImageArray[240:360, :, :, :] = img_arr[340:460, :, :, :]
    testImageArray[:50, :, :, :] = img_arr[120:170, :, :, :]
    testImageArray[50:100, :, :, :] = img_arr[290:340, :, :, :]
    testImageArray[100:150, :, :, :] = img_arr[460:510, :, :, :]
    np.save('/home/yxq/MRI_IMAGE/trainMRI.npy', trainImageArray)
    np.save('/home/yxq/MRI_IMAGE/testMRI.npy', testImageArray)

if __name__ == '__main__':
    load_img()
    
