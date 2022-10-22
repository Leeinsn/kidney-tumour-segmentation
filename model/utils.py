# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import math
import numpy as np
import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindspore import Model, context, Tensor
from mindspore import nn
from mindspore import ops
from mindspore.dataset.transforms.transforms import Compose
from tqdm import tqdm

# from model.dataloader import kits19
from model.model_utils.config import config
# from model.transforms import ExpandChannel, ScaleIntensityRange, OneHot

def correct_nifti_head(img):
    """
    Check nifti object header's format, update the header if needed.
    In the updated image pixdim matches the affine.

    Args:
        img: nifti image object
    """
    dim = img.header["dim"][0]
    if dim >= 5:
        return img
    pixdim = np.asarray(img.header.get_zooms())[:dim]
    norm_affine = np.sqrt(np.sum(np.square(img.affine[:dim, :dim]), 0))
    if np.allclose(pixdim, norm_affine):
        return img
    if hasattr(img, "get_sform"):
        return rectify_header_sform_qform(img)
    return img

def get_random_patch(dims, patch_size, rand_fn=None):
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size`.

    Args:
        dims: shape of source array
        patch_size: shape of patch size to generate
        rand_fn: generate random numbers

    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """
    rand_int = np.random.randint if rand_fn is None else rand_fn.randint
    min_corner = tuple(rand_int(0, ms - ps + 1) if ms > ps else 0 for ms, ps in zip(dims, patch_size))
    return tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default

def _get_scan_interval(image_size, roi_size, num_image_dims, overlap):
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.
    """
    if len(image_size) != num_image_dims:
        raise ValueError("image different from spatial dims.")
    if len(roi_size) != num_image_dims:
        raise ValueError("roi size different from spatial dims.")

    scan_interval = []
    for i in range(num_image_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

def dense_patch_slices(image_size, patch_size, scan_interval):
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval

    Returns:
        a list of slice objects defining each patch
    """
    num_spatial_dims = len(image_size)
    patch_size = patch_size
    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)
    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    return [(slice(None),)*2 + tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]

def create_sliding_window(image, roi_size, overlap):
    num_image_dims = len(image.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")
    image_size_temp = list(image.shape[2:])
    image_size = tuple(max(image_size_temp[i], roi_size[i]) for i in range(num_image_dims))

    scan_interval = _get_scan_interval(image_size, roi_size, num_image_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    windows_sliding = [image[slice] for slice in slices]
    return windows_sliding, slices

def one_hot(labels):
    N, _, D, H, W = labels.shape  # 1*1*64*128*128
    labels = np.reshape(labels, (N, -1)) # 1, 1*64*128*128
    labels = labels.astype(np.int32)
    N, K = labels.shape
    one_hot_encoding = np.zeros((N, config.num_classes, K), dtype=np.float32)
    for i in range(N):
        for j in range(config.num_classes):
            one_hot_encoding[i, j, labels[i]==j] = 1
    labels = np.reshape(one_hot_encoding, (N, config.num_classes, D, H, W))
    return labels

def CalculateDice(y_pred, y, num_classes):
    """
    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        label: ground truth, the first dim is batch.
    """
    # y_pred: 1*2*64*512*512
    y_pred = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)  # 1*1*64*512*512
    intersect = y_pred[y_pred == y]
    
    area_intersect, _ = np.histogram(intersect, bins=num_classes, range=(0, num_classes))
    area_pred_label, _ = np.histogram(y_pred, bins=num_classes, range=(0, num_classes))
    area_label, _ = np.histogram(y, bins=num_classes, range=(0, num_classes))
    area_union = area_pred_label + area_label - area_intersect
    print(area_intersect, area_pred_label, area_label)
    
    dice = 2 * area_intersect / (area_pred_label + area_label)
    # acc = area_intersect / area_label
    
    return dice, None
    

def Patch_Merge(case_pred, label_patch):
    '''
    merge patches into a whole image
    case_pred: 64, 1*num_cls*64*128*128 -> 1*3*64*512*512
    label: 64, 1*1*64*128*128 -> 1*1*64*512*512
    '''
    case = np.zeros((1, config.num_classes, 64, 512, 512))
    label = np.zeros((1, 1, 64, 512, 512))
    case_pred = np.array(case_pred)
    label_patch = np.array(label_patch)
    idx = 0
    
    for x in range(0, 384, 48):
        for y in range(0, 384, 48):
            case[:, :, :, x:x+128, y:y+128] = case_pred[idx]
            label[:, :, :, x:x+128, y:y+128] = label_patch[idx]
            idx += 1
    
    return case, label

def ignore_background(y_pred, label):
    """
    This function is used to remove background (the first channel) for `y_pred` and `y`.
    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        label: ground truth, the first dim is batch.
    """
    label = label[:, 1:] if label.shape[1] > 1 else label
    y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred
    return y_pred, label


class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss = self.network(*inputs)                            # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        self.optimizer(grads)                                   # 使用优化器更新权重参数
        return loss
