import nibabel as nib
import numpy as np
import os
from skimage.transform import resize
from tqdm import tqdm


src_path = '../data'
des_path = '../processed_data/process_step1'
img_name = 'imaging.nii.gz'
seg_name = 'segmentation.nii.gz'

img_save_dir = os.path.join(des_path, 'image')
seg_save_dir = os.path.join(des_path, 'mask')

if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)
if not os.path.exists(seg_save_dir):
    os.makedirs(seg_save_dir)

min_val = -79
max_val = 304
target_size = (128, 248, 248)

case_list = os.listdir(src_path)
case_list.remove('LICENSE')
case_list.remove('kits.json')
case_list = sorted(case_list)[:210]


# just valid value

def get_roi(img, seg):
    # 整理各个轴求和
    stat_z = np.sum(img, axis=(1, 2))
    stat_x = np.sum(img, axis=(0, 2))
    stat_y = np.sum(img, axis=(0, 1))
    # 获取非0坐标
    z_nz = np.nonzero(stat_z)[0]
    x_nz = np.nonzero(stat_x)[0]
    y_nz = np.nonzero(stat_y)[0]
    # 各轴非0开始，结束
    z_sta, z_end = z_nz[0], z_nz[-1]
    x_sta, x_end = x_nz[0], x_nz[-1]
    y_sta, y_end = y_nz[0], y_nz[-1]
    
    return img[z_sta:z_end, x_sta:x_end, y_sta:y_end], seg[z_sta:z_end, x_sta:x_end, y_sta:y_end]

def norm_clip(img, min_val=-79, max_val=304):
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255
    img = np.clip(img, 0, 255) // 1
    img = img.astype('uint8')
    return img


for case in tqdm(case_list):
    img = nib.load(os.path.join(src_path, case, img_name))
    seg = nib.load(os.path.join(src_path, case, seg_name))
    img = img.get_fdata()
    seg = seg.get_fdata()
    img = norm_clip(img)
    img_roi, seg_roi = get_roi(img, seg)
    
    # resize
    img_roi_rs = resize(img_roi, target_size, order=1, mode='constant', cval=0,
       clip=False, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
    seg_roi_rs = resize(seg_roi, target_size, order=0, mode='constant', cval=0,
       clip=False, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
    # 保留seg中标注值
    img_roi_rs = np.round(img_roi_rs)
    seg_roi_rs = np.round(seg_roi_rs)
    
    # 防止四舍五入溢出
    img_roi_rs = np.clip(img_roi_rs, 0, 255)
    
    img_roi_rs = img_roi_rs.astype('uint8')
    seg_roi_rs = seg_roi_rs.astype('uint8')
    
    # 保存得到的数据
    np.save(os.path.join(img_save_dir, case+'.npy'), img_roi_rs)
    np.save(os.path.join(seg_save_dir, case+'.npy'), seg_roi_rs)
    
    print(np.sum(seg_roi_rs==0), np.sum(seg_roi_rs==1), np.sum(seg_roi_rs==2))
    assert np.sum(seg_roi_rs==2) != 0