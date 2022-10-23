import os
import datetime
import mindspore as ms
from mindspore import nn
from mindspore import ops
import mindspore.numpy as np
from mindspore.dataset import GeneratorDataset, vision
from mindspore.dataset.transforms.transforms import Compose
import mindspore.common.dtype as mstype
from mindspore import Tensor, Model, context
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor

from model.dataloader import tumour
from model.transforms_2d import ExpandChannel, RandomHorizontalFlip, RandomVerticalFlip, Normalize, LabelEncode
from model.unet2d import UNet
from model.loss import Dice_CrossEntropy_Loss, Tumour_Dice_CrossEntropy_Loss
from model.lr_schedule import dynamic_lr
from mindspore import Tensor   # , Model, context
from model.utils import CustomTrainOneStepCell
from evaluation.eval_tumour import evaluation

# seg_content = 'tumour'
num_classes = 3
batch_size = 8
lr = 1e-3
warmup_ratio = 0.3
num_epochs = 90
best_tumour_dice = 50
pretrain = True
pretrain_path = './work_dirs/ckpt_20221020/best_tumour_kidney_96_tumour_85.ckpt'

ckpt_save_dir = os.path.join('work_dirs', 'ckpt_tumour_{}'.format(20221020))
log_file = os.path.join(ckpt_save_dir, 'val_log.txt')


if not os.path.exists(ckpt_save_dir):
    os.makedirs(ckpt_save_dir)


net = UNet(in_channel=1, n_class=3)

# 预训练参数
if pretrain:
    param_dict = load_checkpoint(pretrain_path)
    load_param_into_net(net, param_dict)
    print('Load Successfully!')


with open(log_file, 'w') as out:
    print("============== Starting Evaluation ==============")
    total_dice = evaluation(net, num_classes)   
    # https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/train_eval.html

    print("============== End Evaluation ==============")