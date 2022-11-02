# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:39:23 2022

@author: ASUS
"""
import numpy as np
import nibabel as nib
import glob
import os

def Dice(predict, label, eps=1e-6):
    inter = np.sum(predict * label)
    union = np.sum(predict) + np.sum(label)
    dice = (2*inter + eps) / (union + eps)
    return dice

def Hec_dice(predict, label):
    """
    0:其他组织 1:肾脏 2:肿瘤
    hec1:肾脏+肿瘤
    hec2:肿瘤
    """
    hec1 = Dice(predict > 0, label > 0)
    hec2 = Dice(predict == 2, label == 2)
    print(hec1, hec2)
    return (hec1 + hec2) / 2


gt_list = glob.glob(os.path.join('./test', '*', 'segmentation.nii.gz'))
total_dice = []
total_hdice = []
for gt_path in gt_list:
    pred_path = gt_path.replace('./test', './result')
    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()

    # print(np.sum(pred==0), np.sum(pred==1), np.sum(pred==2))
    # pred[pred==2]=1
    # gt[gt==2]=1
    # dice = Dice(pred, gt)
    # print(dice)
    H_dice = Hec_dice(pred, gt)
    # total_dice.append(dice)
    total_hdice.append(H_dice)
    print('H_dice: {}'.format(H_dice))
print('mHecDice: {}'.format(sum(total_hdice)/10))
