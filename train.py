import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import datetime
# import network
import imageio.plugins.freeimage
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from dataset.dataset import Dataset

from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score
import utilities.losses as losses
from utilities.utils import str2bool, count_params
import pandas as pd
from net import Unet, EUNet

# 换模型需要修改的地方
arch_names = list(Unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')

    # 换模型需要修改的地方
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet',
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    # 换数据集需要修改的地方
    parser.add_argument('--dataset', default="LiTS",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    # 换模型需要修改的地方
    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 30)')

    # 换模型需要修改的地方
    parser.add_argument('-b', '--batch-size', default=6, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--deepsupervision', default=False, type=str2bool,
                        help='deep supervision')
    parser.add_argument('--load_model',
                        default="C:/pycode/LITS2017-main1-master/models/epoch233-0.9736-0.8648_model.pth")
    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


ppc = nn.MaxPool2d(kernel_size=2, stride=2)


def train(args, train_loader, model, criterion, optimizer, epoch, criterion2):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    model.train()
    for i, (input, target, distarget) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()  # torch.Size([8, 3, 448, 448])
        target = target.cuda()  # torch.Size([8, 2, 448, 448])
        distarget = distarget.cuda()
        # compute output
        if args.deepsupervision:
            loss = 0
        else:
            output, edge_448 = model(input)
            # distarget = ppc(distarget)

            # print("output shape = ",output.shape)
            loss1 = criterion(output, target)
            # print("edge_224 shape = ",edge_224.shape)
            # print("distarget shape = ",distarget.shape)
            loss2 = criterion2(edge_448, distarget)
            # print("loss1 = ",loss1.item())
            # print("loss2 = ",loss2)
            loss = loss2 + loss1
            iou = iou_score(output, target)
            dice_1 = dice_coef(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices_1s.update(torch.tensor(dice_1), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("loss fin!")

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg)
    ])

    return log


def validate(args, val_loader, model, criterion, criterion2):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target, distarget, filename) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            distarget = distarget.cuda()
            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output, edge_448 = model(input)
                # distarget = ppc(distarget)

                # print("output shape = ",output.shape)
                loss1 = criterion(output, target)
                loss2 = criterion2(edge_448, distarget)
                loss = loss2 + loss1
                iou = iou_score(output, target)
                dice_1 = dice_coef(output, target)
                """
                outimg = torch.sigmoid(output).detach().cpu().numpy()[0][0]
                outimg[outimg>0.5] = 255
                outimg[outimg<0.5] = 0
                outreal = target.detach().cpu().numpy()[0][0]
                outdis = (distarget.detach().cpu().numpy()[0][0])*255
                dout = outreal.copy()
                dout[dout>0] = 133
                dout[outimg>0] = 255
                im = Image.fromarray(np.uint8(dout))
                im1 = Image.fromarray(np.uint8(outdis))
                #im.convert('L').save('/home/luosy/lsycode/liver_seg_pipeline/outputdir/seg/'+filename[0]+'.jpg')
                #im1.convert('L').save('/home/luosy/lsycode/liver_seg_pipeline/outputdir/dismap/' + filename[0] + '.jpg')
                """
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg)
    ])

    return log


def main():
    args = parse_args()
    # args.dataset = "datasets"
    if args.name is None:
        args.name = '%s_%s_lsy' % (args.dataset, args.arch)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('models/{}/{}'.format(args.name, timestamp)):
        os.makedirs('models/{}/{}'.format(args.name, timestamp))

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/{}/{}/args.txt'.format(args.name, timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/{}/{}/args.pkl'.format(args.name, timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss().cuda()
    criterion2 = torch.nn.MSELoss(reduction='mean')
    cudnn.benchmark = True

    # Data loading code
    # img_paths = glob('/home/luosy/lsycode/LITS2017/data512_3/image/*')
    # mask_paths = glob('/home/luosy/lsycode/LITS2017/data512_3/mask/*')
    train_img_paths = glob('E:/LITS/512data/image/*')
    train_mask_paths = glob('E:/LITS/512data/mask/*')
    val_img_paths = glob('E:/LITS/512data/val_image/*')
    val_mask_paths = glob('E:/LITS/512data/val_mask/*')
    # train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #     train_test_split(img_paths, mask_paths, test_size=0.3, random_state=39)
    print("train_num:%s" % str(len(train_img_paths)))
    print("val_num:%s" % str(len(val_img_paths)))

    # create model
    # 换模型需要修改的地方
    print("=> creating model %s" % args.arch)
    # model = Unet.EAU_Net1(args)
    model = EUNet.EAU_Net2(args)
    # model = EUNet.EU2_Net(args)
    # model = network.deeplabv3_resnet50(num_classes=2,output_stride=16)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('E:/liver_seg_pipeline/models/LiTS_Unet_lsy/2021-11-15-09-49-06/0.3822590733772195_model.pth'))
    # model.load_state_dict(torch.load("/content/drive/MyDrive/liver_and_tumor/models/LiTS_Unet_lsy/2021-11-13-12-52-33/0.03782988287905382_model.pth"))
    # /content/drive/MyDrive/liver_and_tumor/models/LiTS_Unet_lsy/2021-11-14-07-50-28/0.5391333386982384_model.pth
    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[36,60,90,120,150,180], gamma=0.6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=20)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',verbose=1,patience=8,cooldown=15,min_lr=5e-7)
    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths, val=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou', 'val_dice'
    ])

    best_loss = 100
    best_dice = -1
    best_iou = 0
    trigger = 0
    first_time = time.time()
    for epoch in range(args.epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('Epoch [%d/%d] | lr = [%f]' % (epoch, args.epochs, lr))
        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch, criterion2)

        # scheduler.step()
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion, criterion2)

        print('loss %.4f - iou %.4f - dice %.4f  - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice_1'], val_log['loss'], val_log['iou'],
                 val_log['dice_1']))

        # print('loss %.4f - iou %.4f - dice %.4f ' %(train_log['loss'], train_log['iou'], train_log['dice']))
        end_time = time.time()
        print("time:", (end_time - first_time) / 60)

        tmp = pd.Series([
            epoch,
            lr,
            train_log['loss'],
            train_log['iou'],
            train_log['dice_1'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice_1'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice_1', 'val_loss', 'val_iou', 'val_dice_1'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/{}/{}/log.csv'.format(args.name, timestamp), index=False)

        trigger += 1

        train_loss = train_log['loss']
        scheduler.step()
        if train_loss < best_loss:
            torch.save(model.state_dict(), 'models/{}/{}/{}_model.pth'.format(args.name, timestamp, train_loss))
            best_loss = train_loss
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    main()