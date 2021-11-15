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
        npimage = npimage[32:480,32:480,:]
        npmask = npmask[32:480,32:480]
        filename = (mask_path.split('/')[-1]).split('.')[0]
        if not self.val:
          flip_num = np.random.randint(0, 8)
          if flip_num == 1:
            npimage = np.flipud(npimage)
            npmask = np.flipud(npmask)
          elif flip_num == 2:
            npimage = np.fliplr(npimage)
            npmask = np.fliplr(npmask)
          elif flip_num == 3:
            npimage = np.rot90(npimage, k=1, axes=(1, 0))
            npmask = np.rot90(npmask, k=1, axes=(1, 0))
          elif flip_num == 4:
            npimage = np.rot90(npimage, k=3, axes=(1, 0))
            npmask = np.rot90(npmask, k=3, axes=(1, 0))
          elif flip_num == 5:
            cropp_img = np.fliplr(npimage)
            cropp_tumor = np.fliplr(npmask)
            npimage = np.rot90(cropp_img, k=1, axes=(1, 0))
            npmask = np.rot90(cropp_tumor, k=1, axes=(1, 0))
          elif flip_num == 6:
            cropp_img = np.fliplr(npimage)
            cropp_tumor = np.fliplr(npmask)
            npimage = np.rot90(cropp_img, k=3, axes=(1, 0))
            npmask = np.rot90(cropp_tumor, k=3, axes=(1, 0))
          elif flip_num == 7:
            cropp_img = np.flipud(npimage)
            cropp_tumor = np.flipud(npmask)
            npimage = np.fliplr(cropp_img)
            npmask = np.fliplr(cropp_tumor)
        # print(filename)
        #a = np.random.randint(0,64)
        # if not self.val:
        #npimage = npimage[32:480,32:480,:]
        #npmask = npmask[32:480,32:480]
        # print("img shape = {} |mask shape = {}".format(npimage.shape,npmask.shape))
        npimage = npimage.transpose((2, 0, 1))

        liver_label = npmask.copy()
        liver_label[npmask == 1] = 0
        liver_label[npmask == 2] = 1

        dismap = npmask.copy()
        dismap[npmask == 1] = 0
        dismap[npmask == 2] = 1
        dismap = ndi.morphology.distance_transform_edt(dismap)
        dismax = dismap.max()
        dismap[dismap>0] /=dismax
        dismap[dismap>0] = 1-dismap[dismap>0]

       # print("mask shape = ",npmask.shape)
        #if not self.val:
        nplabel = np.empty((448,448,1))
        dislabel = np.empty((448, 448, 1))
        # else:
           #  nplabel = np.empty((512, 512, 1))
           #  dislabel = np.empty((512, 512, 1))

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
