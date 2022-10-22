import numpy as np
import os
import nibabel as nib
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Model

from evaluation.test_kidney_tool import get_roi, norm_clip, img_resize, get_slices, preprocess_2d, get_seg_pred, seg_resize, calculate_iou, calculate_dice, get_tumour_roi, get_tumour_roi_data, get_origin_slice_seg
from model.unet2d import UNet


data_path = './data'
tmp_pred_path = './result/kidney'
pred_path = './result/tumour'
ckpt_path = './work_dirs/ckpt_20221020/best_tumour_kidney_96_tumour_85.ckpt'

if not os.path.exists(pred_path):
    os.makedirs(pred_path)

case_list = ['case_00200', 'case_00201', 'case_00202', 'case_00203',
            'case_00204', 'case_00205', 'case_00206', 'case_00207',
            'case_00208', 'case_00209']
img_name = 'imaging.nii.gz'

net = UNet(in_channel=1, n_class=3)
param_dict = load_checkpoint(ckpt_path)
load_param_into_net(net, param_dict)
print('Load Successfully!')

eval_net = Model(net)

for case in case_list:
    # load origin image
    img = nib.load(os.path.join(data_path, case, img_name))
    img = img.get_fdata()
    img = norm_clip(img)
    # load tmp pred
    tmp_pred = np.load(os.path.join(tmp_pred_path, case+'.npy'))
    # get tumour roi
    z_part, x_part, y_part = get_tumour_roi(tmp_pred)
    # print(z_part, x_part, y_part)
    roi_ordi, case_cubic = get_tumour_roi_data(img, z_part, x_part, y_part)
    
    for ordi, cubic in zip(roi_ordi, case_cubic):
        pred_slices = []
        for cubic_slice in cubic:
            slice_shape = cubic_slice.shape
            inp = preprocess_2d(cubic_slice)
            pred = eval_net.predict(inp)
            pred = np.argmax(pred.asnumpy(), axis=1) # pred: 1,1,256,256
            pred = np.squeeze(pred) # 256,256
            # PIL h,w swap
            pred = get_origin_slice_seg(pred.astype('uint8'), (slice_shape[1], slice_shape[0]))
            pred_slices.append(pred)
        pred_slices = np.array(pred_slices)
        z1, z2, x1, x2, y1, y2 = ordi
        tmp_pred[z1:z2, x1:x2, y1:y2] = pred_slices
        # print(ordi)
    # print(np.sum(tmp_pred==0), np.sum(tmp_pred==1), np.sum(tmp_pred==2))
    np.save(os.path.join(pred_path, case+'.npy'), tmp_pred)