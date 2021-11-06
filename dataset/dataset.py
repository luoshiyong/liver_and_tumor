import numpy as np
import cv2  #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color
import scipy.ndimage as ndi
import torch
import torch.utils.data
from torchvision import datasets, models, transforms

np.random.seed(147)
class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths,val=False, transform=None):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.per = 1.0
        self.val = val
    def __len__(self):
        return int(self.per*len(self.img_paths))

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)  # [512,512,3]
        #print("mask_path = ",mask_path)
        npmask = np.load(mask_path)     # [512,512]
        filename = (mask_path.split('\\')[-1]).split('.')[0]
        # print(filename)
        #a = np.random.randint(0,64)
        #if not self.val:
        npimage = npimage[32:480,32:480,:]
        npmask = npmask[32:480,32:480]
        # print("img shape = {} |mask shape = {}".format(npimage.shape,npmask.shape))
        npimage = npimage.transpose((2, 0, 1))

        liver_label = npmask.copy()
        liver_label[npmask == 1] = 1
        liver_label[npmask == 2] = 1

        dismap = npmask.copy()
        dismap[dismap == 1] = 1
        dismap[dismap == 2] = 1
        dismap = ndi.morphology.distance_transform_edt(dismap)
        dismax = dismap.max()
        dismap[dismap>0] /=dismax
        dismap[dismap>0] = 1-dismap[dismap>0]

       # print("mask shape = ",npmask.shape)
        # if not self.val:
        nplabel = np.empty((448,448,1))
        dislabel = np.empty((448, 448, 1))
        # else:
         #   nplabel = np.empty((512, 512, 1))
          #   dislabel = np.empty((512, 512, 1))

        nplabel[:, :, 0] = liver_label
        nplabel = nplabel.transpose((2, 0, 1))

        dislabel[:, :, 0] = dismap
        dislabel = dislabel.transpose((2, 0, 1))

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        dislabel = dislabel.astype('float32')
        # print("npimage shape = ",npimage.shape)
        # print("nplabel shape = ",npmask.shape)
        if not self.val:
            return npimage,nplabel,dislabel
        else:
            return npimage, nplabel, dislabel,filename
