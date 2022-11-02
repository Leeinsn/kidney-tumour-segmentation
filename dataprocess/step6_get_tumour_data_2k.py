import nibabel as nib
import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm


src_path = '../data'
des_path = '../processed_data/process_step3'

img_name = 'imaging.nii.gz'
seg_name = 'segmentation.nii.gz'

img_save_dir = os.path.join(des_path, 'image')
seg_save_dir = os.path.join(des_path, 'mask')

if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)
if not os.path.exists(seg_save_dir):
    os.makedirs(seg_save_dir)
    
def norm_clip(img, min_val=-79, max_val=304):
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255
    img = np.clip(img, 0, 255) // 1
    img = img.astype('uint8')
    return img

def get_roi(img, seg):
    '''
    3d input
    channel reserve
    '''
    assert img.shape == seg.shape
    
    stat_z = np.sum(seg, axis=(1, 2))  # get vector 128
    stat_x = np.sum(seg, axis=(0, 2))  # get vector 248
    stat_y = np.sum(seg, axis=(0, 1))  # get vector 248
    
    z_nonzero = np.nonzero(stat_z)[0]  # get non zero index
    x_nonzero = np.nonzero(stat_x)[0]   
    y_nonzero = np.nonzero(stat_y)[0]
    
    z_part, x_part, y_part = [], [], []
    z_part.append([z_nonzero[0], z_nonzero[-1]])
    x_part.append([x_nonzero[0], x_nonzero[-1]])
    y_part.append([y_nonzero[0], y_nonzero[-1]])
    
    return z_part, x_part, y_part


case_list = os.listdir(src_path)
case_list.remove('LICENSE')
case_list.remove('kits.json')
case_list = sorted(case_list)[200:210]

total_z, total_x, total_y = 0, 0, 0
for case in tqdm(case_list):
    img = nib.load(os.path.join(src_path, case, img_name))
    seg = nib.load(os.path.join(src_path, case, seg_name))
    img = img.get_fdata()
    seg = seg.get_fdata()
    img = norm_clip(img)
    
    assert img.shape == seg.shape
    
    
    z_part, x_part, y_part = get_roi(img, seg)
#     total_z += z_part[0][1] - z_part[0][0]
#     total_x += x_part[0][1] - x_part[0][0]
#     total_y += y_part[0][1] - y_part[0][0]
    print(z_part, x_part, y_part)
# print(total_x/210, total_y/210)
#     idx = 0
#     # extract roi/voi
#     for z1, z2 in z_part:
#         for x1, x2 in x_part:
#             for y1, y2 in y_part:
#                 img_roi = img[z1:z2, x1:x2, y1:y2].astype('uint8')
#                 seg_roi = seg[z1:z2, x1:x2, y1:y2].astype('uint8')
                
#                 # slice along axis z
#                 for z in range(img_roi.shape[0]):
#                     img_roi_2d = img_roi[z].astype('uint8')
#                     seg_roi_2d = seg_roi[z].astype('uint8')
                    
#                     # ignore seg without kidney
#                     if np.max(seg_roi) < 1:
#                         continue

#                     # transfer to PIL
#                     img_roi_2d = Image.fromarray(img_roi_2d)
#                     seg_roi_2d = Image.fromarray(seg_roi_2d)
#                     img_roi_2d = img_roi_2d.resize((384, 384), Image.ANTIALIAS)
#                     seg_roi_2d = seg_roi_2d.resize((384, 384), Image.NEAREST)
#                     # save training data
#                     img_roi_2d.save(os.path.join(img_save_dir, case+'_'+str(idx).zfill(4)+'.png'))
#                     seg_roi_2d.save(os.path.join(seg_save_dir, case+'_'+str(idx).zfill(4)+'.png'))
#                     idx += 1