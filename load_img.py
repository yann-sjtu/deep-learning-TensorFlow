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
        img_arr[n, :, :, :] = sio.loadmat(os.path.join(ad_path,matfile))['Y']
        n += 1
        print(n)
    for matfile in os.listdir(mci_path)[:170]:
        img_arr[n, :, :, :] = sio.loadmat(os.path.join(mci_path,matfile))['Y']
        n += 1
        print(n)
    for matfile in os.listdir(nc_path)[:170]:
        img_arr[n, :, :, :] = sio.loadmat(os.path.join(nc_path,matfile))['Y']
        n += 1
        print(n)
    print("load over")
    np.save('/home/yxq/MRI_IMAGE/DATA_MRI.npy',img_arr)

if __name__ == '__main__':
    load_img()
    
