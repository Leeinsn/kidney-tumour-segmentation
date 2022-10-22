import mindspore.dataset as ds
import numpy as np
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision
from mindspore.dataset.transforms.transforms import Compose
from model.model_utils.config import config
from PIL import Image


class kidney(object):    
    def __init__(self, isTrain=True):
        super(kidney, self).__init__()
        
        with open('./processed_data/process_step1/train_kidney.list') as f:
            train_list = f.readlines()
        f.close()

        with open('./processed_data/process_step1/val_kidney.list') as f:
            val_list = f.readlines()
        f.close()
        
        if isTrain:
            self.data_list = train_list
        else:
            self.data_list = val_list

    def __getitem__(self, idx):
        path = self.data_list[idx].strip().split(' ')
        img = np.load(path[0])
        seg = np.load(path[1])
        return img, seg
        
    def __len__(self):
        return len(self.data_list)
    
    
class tumour(object):
    def __init__(self, isTrain=True):
        super(tumour, self).__init__()
        
        with open('./processed_data/process_step2/train_tumour.list') as f:
            train_list = f.readlines()
        f.close()

        with open('./processed_data/process_step2/val_tumour.list') as f:
            val_list = f.readlines()
        f.close()
        
        if isTrain:
            self.data_list = train_list
        else:
            self.data_list = val_list

    def __getitem__(self, idx):
        path = self.data_list[idx].strip().split(' ')
        img = Image.open(path[0])
        seg = Image.open(path[1])
        return img, seg
        
    def __len__(self):
        return len(self.data_list)