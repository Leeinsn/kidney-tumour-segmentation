from evaluation.test_kidney_tool import *
import glob
import os
import nibabel as nib
import SimpleITK as sitk


input_path = './result'
output_path = './processed_result'

for img_path in glob.glob(os.path.join(input_path, '*', 'segmentation.nii.gz')):
    # img_nib = nib.load(img_path)
    # img = img_nib.get_fdata().astype('uint8')
    # print(img.shape, img.dtype)


    case_name = img_path.split('/')[-2]
    
    img_nii = sitk.ReadImage(img_path, outputPixelType=sitk.sitkUInt8)
    img_fill = sitk.BinaryFillhole(img_nii)
    # print(type(img_fill))
    pro_img = removesmallConnectedCompont(img_fill, 0.1)
    
    # pro_img = nib.Nifti1Image(pro_img, img_nib.affine)
    # pro_img.set_data_dtype(np.uint8)
    
    save_dir = os.path.join(output_path, case_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    sitk.WriteImage(pro_img, os.path.join(save_dir, 'segmentation.nii.gz'))
    # nib.save(pro_img, os.path.join(save_dir, 'segmentation.nii.gz'))