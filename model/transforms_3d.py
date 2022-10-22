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
# =========================================================================

import mindspore as ms
import numpy as np


class ExpandChannel:
    """
    Expand a 1-length channel dimension to the input image.
    """
    def operation(self, data):
        """
        Args:
            data(numpy.array): input data to expand channel.
        """
        return data[None]

    def __call__(self, img, label):
        img_array = self.operation(img)
        seg_array = label
        # seg采用独热编码
        # seg_array = self.operation(label)  
        return img_array, seg_array

    
class ExpandChannelWithLabel:
    """
    Expand a 1-length channel dimension to the input image.
    """
    def operation(self, data):
        """
        Args:
            data(numpy.array): input data to expand channel.
        """
        return data[None]

    def __call__(self, img, label):
        img_array = self.operation(img)
        seg_array = self.operation(label)  
        return img_array, seg_array


class RandomHorizontalFlip:
    '''
    Rotation and Flip
    '''
    def __init__(self, prob=0.5):

        self.prob = prob
    
    def operation(self, data): 
        data = np.flip(data, axis=2).copy()
        return data
    
    def __call__(self, img, label):   # 80, 160, 160
        if np.random.random() > self.prob:
            img = self.operation(img)
            label = self.operation(label)
        return img, label

    
class RandomVerticalFlip:
    '''
    Flip
    '''
    def __init__(self, prob=0.5):

        self.prob = prob
    
    def operation(self, data): 
        data = np.flip(data, axis=1).copy()
        return data
    
    def __call__(self, img, label):   # 80, 160, 160
        if np.random.random() > self.prob:
            img = self.operation(img)
            label = self.operation(label)
        return img, label

    
class GaussianNoise:
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance
    
    def operation(self, img,  noise_variance=(0, 0.1)):
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = np.random.uniform(noise_variance[0], noise_variance[1])
        img = img + np.random.normal(0.0, variance, size=img.shape)
        return img

    
    def __call__(self, img, label):
        if np.random.uniform() < self.prob:
            img = self.operation(img, self.noise_variance)
        return img, label
    
class AjustContrast:
    '''
    single channel contrast augmentation
    '''
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, p_per_sample=0.5):
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
    
    def operation(self, data):
        mn = data.mean()
        if self.preserve_range:
            minm = data.min()
            maxm = data.max()
        if np.random.random() < 0.5 and self.contrast_range[0] < 1:
            factor = np.random.uniform(self.contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])
        data = (data - mn) * factor + mn
        if self.preserve_range:
            data[data < minm] = minm
            data[data > maxm] = maxm
        return data    
        
    def __call__(self, img, label):
        for i in range(img.shape[0]):   # batch_size
            if np.random.uniform() < self.p_per_sample:
                img[i] = self.operation(img[i])
        return img, label
    
class AjustBrightness:
    def __init__(self, mu, sigma, p_per_sample=0.5, p_per_channel=0.5):
        self.p_per_sample = p_per_sample
        self.mu = mu
        self.sigma = sigma
        self.p_per_channel = p_per_channel
        
    def operation(self, data):
        rnd_nb = np.random.normal(self.mu, self.sigma)
        if np.random.uniform() <= self.p_per_channel:
            data += rnd_nb
        return data
        
    def __call__(self, img, label):
        for i in range(img.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                img[i] = sel.operation(img[i])
        return img, label
    
class Normalize:
    '''
    Flip
    '''
    def __init__(self, mean=119.84, std=103.80):

        self.mean = mean
        self.std = std
    
    def operation(self, data): 
        data = (data - self.mean) / self.std 
        return data.astype('float32')
    
    def __call__(self, img, label):   # 256 * 256
        
        img = self.operation(img)
        
        return img, label


class LabelEncode:
    
    def __init__(self, num_classes=3):

        self.num_classes = num_classes
    
    def operation(self, data): 
        
        enc_data = np.zeros((self.num_classes, *data.shape))
        for i in range(self.num_classes):
            enc_data[i][data==i] = 1
        
        return enc_data.astype('float32')
    
    def __call__(self, img, label):   
        
        label = self.operation(label)
        
        return img, label
