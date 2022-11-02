from model.dataloader import kidney
from model.unet3d import UNet3d_
from model.loss import Dice_CrossEntropy_Loss
from model.lr_schedule import dynamic_lr
from model.transforms_3d import Normalize, LabelEncode, ExpandChannel, RandomHorizontalFlip, RandomVerticalFlip
from model.utils import CustomTrainOneStepCell
from evaluation.eval_kidney import evaluation

import os
import datetime
import mindspore as ms
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms.transforms import Compose
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net


batch_size = 4
num_classes = 2
lr = 0.001
warmup_ratio = 0.3
num_epochs = 90
pretrain = True
pretrain_path = './work_dirs/ckpt_kidney_20221021/best_kidney.ckpt'

best_dice = 0
ckpt_save_dir = os.path.join('work_dirs', 'ckpt_kidney_{}'.format(20221021))
log_file = os.path.join(ckpt_save_dir, 'val_log.txt')

if not os.path.exists(ckpt_save_dir):
    os.makedirs(ckpt_save_dir)

net = UNet3d_(num_classes)

if pretrain:
    param_dict = load_checkpoint(pretrain_path)
    load_param_into_net(net, param_dict)
    print('Load Successfully!')


with open(log_file, 'w') as out:
    print("============== Starting Evaluation ==============")
    total_dice = evaluation(net, num_classes, out)   # https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/train_eval.html

    print("============== End Evaluation ==============")
