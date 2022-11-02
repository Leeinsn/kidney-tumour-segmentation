import nibabel as nib
import numpy as np
import glob
import argparse
import os
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from tqdm import tqdm
from mindspore import Model

from model.unet3d import UNet3d_
# from model.unet2d import UNet
from model.unet3plus import UNet3Plus

from evaluation.inference_hooks import get_kidney_pred, get_tumour_pred, get_tumour_pred_test
from evaluation.test_kidney_tool import removesmallConnectedCompont, BinaryFillhole


parser = argparse.ArgumentParser(description='Input path and output path.')

parser.add_argument('--input_path', type=str, help='Input Path')
parser.add_argument('--output_path', type=str, help='Output Path')
parser.add_argument('--weights_1', type=str, help='Weights of UNet-3D')
parser.add_argument('--weights_2', type=str, help='Weights of UNet')

args = parser.parse_args()

print(args)

input_path = args.input_path
output_path = args.output_path
ckpt_path_1 = args.weights_1
ckpt_path_2 = args.weights_2


# 测试输入图像必须以imaging.nii.gz命名
test_list = sorted(glob.glob(os.path.join(input_path, '*', 'imaging.nii.gz')))
# print(test_list)
# network prepare
net_kid = UNet3d_(2)
param_dict = load_checkpoint(ckpt_path_1)
load_param_into_net(net_kid, param_dict)
# print('Load Successfully!')
eval_net_kid = Model(net_kid)

# net_tum = UNet(in_channel=1, n_class=3)
net_tum = UNet3Plus(in_channels=1, n_classes=3)
param_dict = load_checkpoint(ckpt_path_2)
load_param_into_net(net_tum, param_dict)
# print('Load Successfully!')
eval_net_tum = Model(net_tum)

for img_path in tqdm(test_list):
    
    img_nib = nib.load(img_path)
    img = img_nib.get_fdata()
    
    kidney_pred = get_kidney_pred(img, eval_net_kid)
    kidney_pred = BinaryFillhole(kidney_pred)

    kidney_pred = removesmallConnectedCompont(kidney_pred, 0.2)
    tumour_pred = get_tumour_pred(img, kidney_pred, eval_net_tum)
    tumour_only = tumour_pred.copy()
    tumour_only[tumour_only!=2] = 0
    tumour_only[tumour_only==2] = 1
    tumour_only = BinaryFillhole(tumour_only)
    tumour_only = removesmallConnectedCompont(tumour_only, 0.2)
    tumour_pred[tumour_pred==2] = 1
    tumour_pred[tumour_only==1] = 2

    pred = nib.Nifti1Image(tumour_pred, img_nib.affine)
    pred.set_data_dtype(np.uint8)
    
    case_name = img_path.split('/')[-2]
    
    save_dir = os.path.join(output_path, case_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nib.save(pred, os.path.join(save_dir, 'segmentation.nii.gz'))
    
    # file_name = case_name.replace('case', 'prediction')
    # nib.save(pred, os.path.join(output_path, file_name+'.nii.gz'))

