from model.dataloader import kidney
from model.transforms_3d import Normalize, LabelEncode, ExpandChannel

from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms.transforms import Compose
import math



import numpy as np
import os
import nibabel as nib
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Model
from tqdm import tqdm

from evaluation.test_kidney_tool import get_roi, norm_clip, img_resize, get_slices, preprocess, get_seg_pred, seg_resize, calculate_iou, calculate_dice
from model.unet3d import UNet3d_


num_classes = 2
ckpt_path = './work_dirs/ckpt_kidney_20221021/best_kidney.ckpt'

eval_dataset = kidney(isTrain=False)
eval_dataset = GeneratorDataset(source=eval_dataset, column_names=['img', 'label'], shuffle=False)
trans = Compose([Normalize(mean=119.84, std=103.80),
                 ExpandChannel(),
            ])
eval_dataset = eval_dataset.map(operations=trans, input_columns=["img", "label"])
eval_dataset = eval_dataset.batch(1)

eval_data_size = eval_dataset.get_dataset_size()
print("val dataset length is:", eval_data_size)

net = UNet3d_(2)
param_dict = load_checkpoint(ckpt_path)
load_param_into_net(net, param_dict)
print('Load Successfully!')
eval_net = Model(net)


total_area_intersect = np.zeros(num_classes)
total_area_pred_label = np.zeros(num_classes)
total_area_label = np.zeros(num_classes)

idx = 1
pred_slices, label_slices = [], []

for batch in tqdm(eval_dataset.create_dict_iterator(num_epochs=1), total=eval_data_size):
    image = batch["img"]  # 1*1*256*256
    label = batch["label"]  # 1*1*256*256
    # print("current image shape is {}".format(image.shape), flush=True)
    pred = eval_net.predict(image)
    pred = pred.asnumpy()
    label = label.asnumpy()
    pred = np.argmax(pred, axis=1) # pred: 1,1,80,160,160
    pred = np.squeeze(pred) # 80*160*160
    label = np.squeeze(label)
    pred_slices.append(pred)
    label_slices.append(label)
    
    if idx % 18 == 0:
        seg_pred = get_seg_pred(pred_slices)
        seg = get_seg_pred(label_slices)
        
        area_intersect, area_pred_label, area_label = calculate_iou(seg_pred, seg)
    
        total_area_intersect += area_intersect
        total_area_pred_label += area_pred_label
        total_area_label += area_label
        idx = 0
    idx += 1
    
dice = calculate_dice(total_area_intersect, total_area_pred_label, total_area_label)
print(dice)
print("mean Dice on case 200 ~ 209 is")

print("    background: {:.2f}, kidney: {:.2f} ".format(dice[0]*100, dice[1]*100))