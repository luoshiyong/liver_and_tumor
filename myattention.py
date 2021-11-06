import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
"""
class AttentionBlock(nn.Module):
    def __init__(self,inch,size):
        super(AttentionBlock, self).__init__()
        self.conv33 = nn.Conv3d(1,1,3,padding=1)
        self.conv11 = nn.Conv2d(inch,1,1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weights = nn.Sequential(nn.Linear(inch, int(inch/2)),
                                     nn.LeakyReLU(0.05),
                                     nn.Linear(int(inch/2), inch),
                                     nn.Sigmoid()
        )
        self.ac = nn.Sigmoid()
        self.outconv = nn.Conv2d(2*inch,inch,1)
        self.size = size
        self.inch = inch
        # self.glbalpool = F.avg_pool2d(x, kernel_size=(size, 448), padding=0)
    def forward(self,x):
        bs, ch,ww,hh = x.size()
        xy = self.conv11(x)      # [1, 1, 28, 28]
        re_weights = self.pool(x).view(bs,ch) # [1, 256, 1, 1]
        re_weights = self.weights(re_weights).view(bs, ch, 1, 1)
        xy = xy.repeat(1,self.inch,1,1)
        re_data = xy*re_weights
        re_data = self.conv33(re_data.view(bs,1,ch,ww,hh))
        out = x*self.ac(re_data.view(bs,ch,ww,hh))
        out = self.outconv(torch.cat([x,out],dim=1))
        # print("xy shape = ", xy.shape)
        # print("h shape = ", re_weights.shape)
        return out
input = torch.randn(4,512,28,28)
model = AttentionBlock(512,28)
out = model(input)
print(out.shape)
"""
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from collections import Counter
from sklearn import preprocessing
path = "C:/pycode/LITS2017-main1-master/data/trainMask_k1_1217/3_50.npy"
mask = np.load(path)
liver =mask.copy()
liver[liver>0] = 1
tumor =mask.copy()
tumor[tumor==1] = 0
tumor[tumor==2] = 1
boundary_l = cv2.Canny(liver,0,1)
plt.imshow(boundary_l)
plt.show()
boundary_lesion = ndi.morphology.distance_transform_edt(tumor)
boundary_liver  = ndi.morphology.distance_transform_edt(liver)
max = boundary_liver.max()
boundary_liver[boundary_liver>0] /=max
boundary_liver[boundary_liver>0] = 1-boundary_liver[boundary_liver>0]

max2 = boundary_lesion.max()
boundary_lesion[boundary_lesion>0] /= max2
boundary_lesion[boundary_lesion>0] = 1-boundary_lesion[boundary_lesion>0]
# boundary_liver = 1 - boundary_liver
# boundary_liver[boundary_liver==1.0] = 0
# min_max_scaler = preprocessing.MinMaxScaler()
# boundary_liver = min_max_scaler.fit_transform(boundary_liver)
#boundary_liver[boundary_liver>0] = 1-boundary_liver
print("mask shape = ",mask.shape)
dd = Counter(boundary_liver.flatten())
print(dd)
plt.imshow(mask)
plt.show()
plt.imshow(boundary_liver)
plt.show()
plt.imshow(boundary_lesion)
plt.show()
"""
"""
import network
import torch
import torch.nn as nn
model = network.deeplabv3_resnet50(num_classes=2,output_stride=16)
model = torch.nn.DataParallel(model)
input = torch.randn(4,3,512,512)
#input = torch.randn(4,3,448,448)
out = model(input)
print(out.shape)
"""
import torch.backends.cudnn as cudnn
import torchvision
import ttach as tta

import network
from dataset.dataset import Dataset

from net import Unet
from utilities.utils import str2bool, count_params
import joblib
import imageio
#import ttach as tta

import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from skimage import measure
from scipy.ndimage import label

import glob
from time import time

import copy
import math
import argparse
import random
import warnings
import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
test_ct_path = "C:/Users/luosy/Desktop/lits_res/all_slice_seg"
pptk = 0
for file_index, file in enumerate(os.listdir(test_ct_path)):
    start = time()

    res = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
    res_array = sitk.GetArrayFromImage(res)
    res_array[:,32:480,32:480] = 0
    ctpp = Counter(res_array.flatten())
    # print(ctpp)
    if ctpp[1]!=0:
        print(file)
        pptk+=1
        print(ctpp[1])
    #break
print("non 0 = ",pptk)