import numpy as np
from skimage.transform import resize
import mindspore as ms
import math
from PIL import Image


def get_roi(img, seg=None):
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
    
    roi_range = [z_sta, z_end, x_sta, x_end, y_sta, y_end]
    
    if seg is None:
        return roi_range, img[z_sta:z_end, x_sta:x_end, y_sta:y_end]
    else:
        return roi_range, img[z_sta:z_end, x_sta:x_end, y_sta:y_end], seg[z_sta:z_end, x_sta:x_end, y_sta:y_end]

def norm_clip(img, min_val=-79, max_val=304):
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255
    img = np.clip(img, 0, 255) // 1
    img = img.astype('uint8')
    return img

def img_resize(img, target_size):
    '''
    img must be 3d ndarray
    '''
    img_rs = resize(img, target_size, order=1, mode='constant', cval=0,
       clip=False, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
    img_rs = np.round(img_rs)
    img_rs = np.clip(img_rs, 0, 255)
    img_rs = img_rs.astype('uint8')
    
    assert img_rs.shape == target_size
    
    return img_rs

def seg_resize(seg, target_size):
    '''
    seg must be 3d ndarray
    '''
    seg_rs = resize(seg, target_size, order=0, mode='constant', cval=0,
       clip=False, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
    seg_rs = np.round(seg_rs)
    seg_rs = seg_rs.astype('uint8')
    
    assert seg_rs.shape == target_size
    
    return seg_rs


def get_slices(img):
    
    assert img.shape == (128, 248, 248)
    
    img_slices = []
    for z_sta in [0, 48]:
        for x_sta in [0, 44, 88]:
            for y_sta in [0, 44, 88]:
                img_slice = img[z_sta:z_sta+80, x_sta:x_sta+160, y_sta:y_sta+160]
                img_slices.append(img_slice)
    return img_slices


def get_seg_pred(pred_slices):
    
    seg_roi_pred = np.zeros((128, 248, 248))
    
    idx = 0
    for z_sta in [0, 48]:
        for x_sta in [0, 44, 88]:
            for y_sta in [0, 44, 88]:
                seg_roi_pred[z_sta:z_sta+80, x_sta:x_sta+160, y_sta:y_sta+160] = pred_slices[idx]
                idx += 1
    return seg_roi_pred


def preprocess(img, mean=119.84, std=103.80):
    '''
    preprocess
    '''
    img = (img - mean) / std
    img = img[None, None]
    img = img.astype('float32')
    img = ms.Tensor(img)
    
    return img


def preprocess_2d(img, mean=119.84, std=103.80):
    '''
    preprocess
    '''
    img = Image.fromarray(img)
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = np.array(img)
    img = (img - mean) / std
    img = img[None, None]
    img = img.astype('float32')
    img = ms.Tensor(img)
    
    return img


def get_origin_slice_seg(pred, shape):
    
    pred = Image.fromarray(pred)
    pred = pred.resize(shape, Image.NEAREST)
    pred = np.array(pred)
    
    return pred


def calculate_iou(y_pred, y, num_classes=2):
    """
    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        label: ground truth, the first dim is batch.
    """
    # y_pred = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
    intersect = y_pred[y_pred == y]
    
    area_intersect, _ = np.histogram(intersect, bins=num_classes, range=(0, num_classes))
    area_pred_label, _ = np.histogram(y_pred, bins=num_classes, range=(0, num_classes))
    area_label, _ = np.histogram(y, bins=num_classes, range=(0, num_classes))
    # area_union = area_pred_label + area_label - area_intersect
    # print(area_intersect, area_pred_label, area_label)
    
    return area_intersect, area_pred_label, area_label


def calculate_dice(total_area_intersect, total_area_pred_label, total_area_label):
    dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
    # acc = area_intersect / area_label
    for idx, ele in enumerate(dice):
        if math.isnan(ele):
            dice[idx] = 0
    return dice


def get_tumour_roi(seg):
    '''
    3d input
    channel reserve
    '''
    # assert img.shape == seg.shape
    
    stat_z = np.sum(seg, axis=(1, 2))  # get vector 128
    stat_x = np.sum(seg, axis=(0, 2))  # get vector 248
    stat_y = np.sum(seg, axis=(0, 1))  # get vector 248
    # print(stat_y)
    z_nonzero = np.nonzero(stat_z)[0]  # get non zero index
    x_nonzero = np.nonzero(stat_x)[0]   
    y_nonzero = np.nonzero(stat_y)[0]
    # print(y_nonzero)
    # discontinuity threshold
    dc_th = 15
    
    z_con = np.where(np.diff(z_nonzero)>dc_th)[0]
    x_con = np.where(np.diff(x_nonzero)>dc_th)[0]   # get difference 差分
    y_con = np.where(np.diff(y_nonzero)>dc_th)[0]   # 返回的断点为第一段的末索引
    
    # print(z_con, x_con, y_con)
    z_part, x_part, y_part = [], [], []
    
    if z_con.size == 0:
        z_part.append([z_nonzero[0], z_nonzero[-1]])
    elif z_con.size == 1:
        z_part.append([z_nonzero[0], z_nonzero[z_con[0]]])
        z_part.append([z_nonzero[z_con[0]+1], z_nonzero[-1]])
    else:
        print('Data Error! Please Check X!')
    
    if x_con.size == 0:
        x_part.append([x_nonzero[0], x_nonzero[-1]])
    elif x_con.size == 1:
        x_part.append([x_nonzero[0], x_nonzero[x_con[0]]])
        x_part.append([x_nonzero[x_con[0]+1], x_nonzero[-1]])
    else:
        print('Data Error! Please Check X!')
    
    if y_con.size == 0:
        y_part.append([y_nonzero[0], y_nonzero[-1]])
    elif y_con.size == 1:
        y_part.append([y_nonzero[0], y_nonzero[y_con[0]]])
        y_part.append([y_nonzero[y_con[0]+1], y_nonzero[-1]])
    else:
        # print('Data Error! Please Check Y!')
        y_part.append([y_nonzero[0], y_nonzero[y_con[0]]])
        y_part.append([y_nonzero[y_con[-1]+1], y_nonzero[-1]])
    
    # return slice index per VOI
    return z_part, x_part, y_part


def get_tumour_roi_data(img, z_part, x_part, y_part):
    roi_ordi = []
    case_cubic = []

    for z1, z2 in z_part:
        if z1 == z2:
            continue
        for x1, x2 in x_part:
            if x1 == x2:
                continue
            for y1, y2 in y_part:
                if y1 == y2:
                    continue
                roi_ordi.append([z1, z2, x1, x2, y1, y2])
                img_roi = img[z1:z2, x1:x2, y1:y2].astype('uint8')
                case_cubic.append(img_roi)
    return roi_ordi, case_cubic