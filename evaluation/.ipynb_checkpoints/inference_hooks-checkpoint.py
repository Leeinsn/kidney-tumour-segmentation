import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from evaluation.test_kidney_tool import *
# (get_roi, norm_clip, img_resize, get_slices, preprocess, get_seg_pred, seg_resize, calculate_iou, calculate_dice, preprocess_2d, seg_resize, get_tumour_roi, get_tumour_roi_data, get_origin_slice_seg)
from model.unet3d import UNet3d_
from mindspore import Model
# from evaluation.test_kidney_tool import get_roi, norm_clip, img_resize, get_slices


def get_kidney_pred(img, eval_net_kid, num_classes=2, input_size=(80, 160, 160),
                    target_size=(128, 248, 248)):
    '''
    img -> ndarray: z, x, y
    '''
    # print('loading nibabel successfully!')
    # original size
    origin_size = img.shape
    # value preprocess
    img = norm_clip(img)
    # get roi area
    roi, img_roi = get_roi(img)
    # roi size
    roi_size = (roi[1]-roi[0], roi[3]-roi[2], roi[5]-roi[4])
    
    # resize to suitable size
    img_roi_rs = img_resize(img_roi, target_size)
    # print('resize successfully!')
    # print(img_roi_rs.shape)
    # slices  128*248*248 -> 18*80*160*160
    img_slices = get_slices(img_roi_rs)
    
    pred_slices = []
    
    for img_slice in img_slices:
        inp = preprocess(img_slice) # 1*1*80*160*160
        pred = eval_net_kid.predict(inp) # 1*2*80*160*160
        # pred = np.squeeze(pred) # 2*80*160*160
        pred = np.transpose(pred.asnumpy(), (0,2,3,4,1))  # 1*80*160*160*2
        pred = softmax(pred)
        pred_slices.append(pred)
    # pred_slices: 18*80*160*160*2
    pred = get_seg_pred(pred_slices)  # 128*248*248*2
    pred = np.argmax(pred, axis=-1)
    # print(pred.shape)
    # pred size -> origin size
    # print(np.sum(pred==0), np.sum(pred==1), np.sum(pred==2))
    # print(pred.shape, roi_size)
    pred = seg_resize(pred, roi_size)
    # seg pred component
    seg_pred = np.zeros(origin_size, dtype='uint8')
    seg_pred[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]] = pred
    # print(np.sum(seg_pred==0), np.sum(seg_pred==1), np.sum(seg_pred==2))
    
    return seg_pred


def get_tumour_pred(img, kidney_pred, eval_net_tum):
    '''
    img -> ndarray: z, x, y
    kidney_pred -> ndarray: z, x, y
    '''
    # preprocess
    img = norm_clip(img)
    # get tumour roi
    z_part, x_part, y_part = get_tumour_roi(kidney_pred)
    # z_part, x_part, y_part = get_roi_2k(kidney_pred)
    # print(z_part, x_part, y_part)
    
    roi_ordi, case_cubic = get_tumour_roi_data(img, z_part, x_part, y_part)
    
    for ordi, cubic in zip(roi_ordi, case_cubic):
        pred_slices = []
        for cubic_slice in cubic:
            slice_shape = cubic_slice.shape
            inp = preprocess_2d(cubic_slice, size=(256, 256))
            pred = eval_net_tum.predict(inp)
            pred = np.argmax(pred.asnumpy(), axis=1) # pred: 1,1,256,256
            pred = np.squeeze(pred) # 256,256
            # PIL h,w swap
            pred = get_origin_slice_seg(pred.astype('uint8'), (slice_shape[1], slice_shape[0]))
            pred_slices.append(pred)
        pred_slices = np.array(pred_slices)
        z1, z2, x1, x2, y1, y2 = ordi
        kidney_pred[z1:z2, x1:x2, y1:y2] = pred_slices
    
    return kidney_pred


def get_tumour_pred_test(img, eval_net_tum):
    # preprocess
    img = norm_clip(img)
    pred_slices = []
    for cubic_slice in img:
        inp = preprocess_2d(cubic_slice, size=(512, 512))
        pred = eval_net_tum.predict(inp)
        pred = np.argmax(pred.asnumpy(), axis=1) # pred: 1,1,256,256
        pred = np.squeeze(pred) # 256,256
        pred = pred.astype('uint8')
        pred_slices.append(pred)
    pred_slices = np.array(pred_slices)
    
    return pred_slices