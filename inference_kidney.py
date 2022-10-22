import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from evaluation.test_kidney_tool import get_roi, norm_clip, img_resize, get_slices, preprocess, get_seg_pred, seg_resize, calculate_iou, calculate_dice
from model.unet3d import UNet3d_
from mindspore import Model


data_path = './data'
pred_path = './result/kidney'

if not os.path.exists(pred_path):
    os.makedirs(pred_path)

case_list = ['case_00200', 'case_00201', 'case_00202', 'case_00203',
            'case_00204', 'case_00205', 'case_00206', 'case_00207',
            'case_00208', 'case_00209']
img_name = 'imaging.nii.gz'
ckpt_path = './work_dirs/ckpt_kidney_20221021/best_kidney.ckpt'

num_classes = 2
# inference data size
input_size = (80, 160, 160)
# cubic target size
target_size = (128, 248, 248)

# network prepare
net = UNet3d_(2)
param_dict = load_checkpoint(ckpt_path)
load_param_into_net(net, param_dict)
print('Load Successfully!')
eval_net = Model(net)

for case in tqdm(case_list):
    print(case)
    img = nib.load(os.path.join(data_path, case, img_name))
    img = img.get_fdata()
    print('loading nibabel successfully!')
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
    print('resize successfully!')
    print(img_roi_rs.shape)
    # slices  128*248*248 -> 18*80*160*160
    img_slices = get_slices(img_roi_rs)
    
    pred_slices = []
    
    for img_slice in img_slices:
        inp = preprocess(img_slice)
        pred = eval_net.predict(inp)
        pred = np.argmax(pred.asnumpy(), axis=1) # pred: 1,1,80,160,160
        pred = np.squeeze(pred) # 80*160*160
        pred_slices.append(pred)
    
    pred = get_seg_pred(pred_slices)  # 128*248*248
    # pred size -> origin size
    pred = seg_resize(pred, roi_size)
    # seg pred component
    seg_pred = np.zeros(origin_size, dtype='uint8')
    seg_pred[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]] = pred
    
    np.save(os.path.join(pred_path, case+'.npy'), seg_pred)