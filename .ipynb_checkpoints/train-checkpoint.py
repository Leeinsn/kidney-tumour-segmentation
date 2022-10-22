from model.dataloader import kits19
from model.unet3d import UNet3d_
from model.vnet import VNet
from model.loss import Dice_CrossEntropy_Loss, Tumour_Dice_CrossEntropy_Loss
from model.lr_schedule import dynamic_lr
from model.model_utils.config import config
from model.transforms import GaussianNoise, ScaleIntensityRange, OneHot, RandomRotFlip, AjustContrast
from model.utils import CustomTrainOneStepCell, evaluation

import os
import datetime
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms.transforms import Compose
import mindspore.common.dtype as mstype
from mindspore import Tensor, Model, context
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor


# ckpt_path = './work_dirs/checkpoint/ckpt_20221008/best_kidney.ckpt'
seg_content = 'tumour'
num_classes = 3

train_dataset = kits19(seg_content)
train_dataset = GeneratorDataset(source=train_dataset, column_names=['img', 'label'],
                          num_parallel_workers=1, shuffle=True)
train_dataset = train_dataset.batch(config.tumour_batch_size)
# 68.02, 63.81
trans = Compose([ScaleIntensityRange(src_mean=119.84, src_std=103.80, is_clip=False),
                 # RandomRotFlip(),
                 # AjustContrast(),
                 # GaussianNoise(),
                 OneHot(num_classes)])

train_dataset = train_dataset.map(operations=trans, input_columns=["img", "label"], num_parallel_workers=1)

train_data_size = train_dataset.get_dataset_size()
print("train dataset length is:", train_data_size)

# net = UNet3d_(num_classes)
net = VNet()

loss = Tumour_Dice_CrossEntropy_Loss()
lr = Tensor(dynamic_lr(config, train_data_size, config.tumour_epoch_size), mstype.float32)
optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr)


# 预训练参数
# param_dict = load_checkpoint(ckpt_path)
# load_param_into_net(net, param_dict)

# 是否采用半精度
if config.device_target == 'GPU' and config.enable_fp16_gpu:
    print('Error! Method is not implement!')
else:
    net_with_loss = nn.WithLossCell(net, loss)
    
# 定义训练网络，封装网络和优化器
train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
train_net.set_train()

# 真正训练迭代过程
step = 0
steps = train_data_size
best_dice = config.best_dice
epoch_size = config.tumour_epoch_size
cast = ops.Cast()

for epoch in range(epoch_size):
    for d in train_dataset.create_dict_iterator():
        d['img'] = cast(d['img'], ms.float32)
        l = train_net(d["img"], d["label"])
        if step % 20 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{time}, "
                  f"Epoch: [{epoch} / {epoch_size}], "
                  f"step: [{step} / {steps * epoch_size}], "
                  f"loss: {l}, "
                  f"lr: {lr[step]}")
        step = step + 1

    print("============== Starting Evaluation ==============")
    total_dice = evaluation(net, loss, seg_content)   # https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/train_eval.html

    print("============== End Evaluation ==============")

    if total_dice[2] > best_dice[1]:
        best_dice[1] = total_dice[2]
        ckpt_save_dir = os.path.join(config.output_path,
                                        config.checkpoint_path,
                                        'ckpt_{}'.format(config.ckpt_data))
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
        ms.save_checkpoint(net, 
                           os.path.join(ckpt_save_dir, 'best_tumour.ckpt'))
        print('checkpoint is saved successfully!')
