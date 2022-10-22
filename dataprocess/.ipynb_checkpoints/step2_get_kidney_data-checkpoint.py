import os
import glob
import numpy as np


src_img_path = '../processed_data/process_step1/image'
src_seg_path = '../processed_data/process_step1/mask'
des_img_path = '../processed_data/process_step1/image_kidney'
des_seg_path = '../processed_data/process_step1/mask_kidney'

if not os.path.exists(des_img_path):
    os.makedirs(des_img_path)
if not os.path.exists(des_seg_path):
    os.makedirs(des_seg_path)

# train_data size
train_size = (80, 160, 160)

src_img_list = sorted(glob.glob(os.path.join(src_img_path, '*.npy')))
src_seg_list = sorted(glob.glob(os.path.join(src_seg_path, '*.npy')))

def get_slices(img, seg):
    # 128 * 248 * 248 -> 80 * 160 * 160
    assert img.shape == seg.shape
    
    img_slices = []
    seg_slices = []
    
    # cut into 18 cubes
    for z_sta in [0, 48]:
        for x_sta in [0, 44, 88]:
            for y_sta in [0, 44, 88]:
                img_slice = img[z_sta:z_sta+80, x_sta:x_sta+160, y_sta:y_sta+160]
                seg_slice = seg[z_sta:z_sta+80, x_sta:x_sta+160, y_sta:y_sta+160]
                img_slices.append(img_slice)
                seg_slices.append(seg_slice)
                
    return img_slices, seg_slices

for case_idx, (img_path, seg_path) in enumerate(zip(src_img_list, src_seg_list)):
    img = np.load(img_path)
    seg = np.load(seg_path)
    # generate kidney mask
    seg[seg==2] = 1
    
    img_slices, seg_slices = get_slices(img, seg)
    
    # save slices
    for slice_idx, (img_slice, seg_slice) in enumerate(zip(img_slices, seg_slices)):
        np.save(os.path.join(des_img_path,
            'case_'+str(case_idx).zfill(5)+'_'+str(slice_idx).zfill(3)+'.npy'), img_slice)
        np.save(os.path.join(des_seg_path,
            'case_'+str(case_idx).zfill(5)+'_'+str(slice_idx).zfill(3)+'.npy'), seg_slice)