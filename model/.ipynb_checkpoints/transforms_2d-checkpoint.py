import numpy as np
import mindspore as ms


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
        data = np.flip(data, axis=1).copy()
        return data
    
    def __call__(self, img, label):   # 256 * 256
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
        data = np.flip(data, axis=0).copy()
        return data
    
    def __call__(self, img, label):   # 256 * 256
        if np.random.random() > self.prob:
            img = self.operation(img)
            label = self.operation(label)
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
    
    def __call__(self, img, label):   # 256 * 256
        
        label = self.operation(label)
        
        return img, label
