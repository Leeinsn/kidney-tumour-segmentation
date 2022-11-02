import os
import glob
from PIL import Image
import numpy as np

img_path = '../processed_data/process_step3/image'
seg_path = '../processed_data/process_step3/mask'

img_des_path = './processed_data/process_step3/image'
seg_des_path = './processed_data/process_step3/mask'

img_list = sorted(os.listdir(img_path))
seg_list = sorted(os.listdir(seg_path))
# if Error, please check is the directory consists .ipynb_checkpoints 
assert len(img_list) == len(seg_list)

train_list = []
val_list = []
for img_name, seg_name in zip(img_list, seg_list):
    seg = np.array(Image.open(os.path.join(seg_path, seg_name)))
    # 剔除训练集负例
    if np.sum(seg) ==0:
        continue
    if img_name.split('_')[1] < '00200':
        train_list.append([os.path.join(img_des_path, img_name), 
                         os.path.join(seg_des_path, seg_name)])
    else:
        val_list.append([os.path.join(img_des_path, img_name), 
                         os.path.join(seg_des_path, seg_name)])  
        
with open('../processed_data/process_step3/train_tumour.list', 'w') as f:
    for item in train_list:
        f.write(item[0]+' '+item[1]+'\n')
    f.close()
    
with open('../processed_data/process_step3/val_tumour.list', 'w') as f:
    for item in val_list:
        f.write(item[0]+' '+item[1]+'\n')
    f.close()