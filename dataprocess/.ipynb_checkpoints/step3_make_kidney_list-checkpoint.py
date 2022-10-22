import os
import glob

file_path = '../processed_data/process_step1'
read_path = './processed_data/process_step1'
img_path = os.path.join(file_path, 'image_kidney')
seg_path = os.path.join(file_path, 'mask_kidney')

train_list = []
val_list = []

# split
for idx, item_name in enumerate(sorted(os.listdir(img_path))):
    if idx < 3600:
        train_list.append(item_name)
    else:
        val_list.append(item_name)

# train list
with open('../processed_data/process_step1/train_kidney.list', 'w') as f:
    for item in train_list:    
        # 全路径
        img = os.path.join(read_path, 'image_kidney', item)
        seg = os.path.join(read_path, 'mask_kidney', item)
        f.write(img+' '+seg+'\n')
    f.close()
    
# val list
with open('../processed_data/process_step1/val_kidney.list', 'w') as f:
    for item in val_list:    
        # 全路径
        img = os.path.join(read_path, 'image_kidney', item)
        seg = os.path.join(read_path, 'mask_kidney', item)
        f.write(img+' '+seg+'\n')
    f.close()