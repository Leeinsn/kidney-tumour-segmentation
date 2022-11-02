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
from model.unet3plus import UNet3Plus
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
pretrain = False
pretrain_path = './work_dirs/ckpt_20221020/best_tumour_kidney_96_tumour_85.ckpt'

ckpt_save_dir = os.path.join('work_dirs', 'ckpt_unet3plus_{}'.format(20221102))
log_file = os.path.join(ckpt_save_dir, 'tumour_log.txt')


if not os.path.exists(ckpt_save_dir):
    os.makedirs(ckpt_save_dir)

train_dataset = tumour()
train_dataset = GeneratorDataset(source=train_dataset, column_names=['img', 'label'],
                          num_parallel_workers=1, shuffle=True)

trans = Compose([RandomVerticalFlip(),
                 RandomHorizontalFlip(),
                 Normalize(),
                 LabelEncode(num_classes),
                 ExpandChannel(),
                ])

train_dataset = train_dataset.map(operations=trans, input_columns=["img", "label"])

train_dataset = train_dataset.batch(batch_size)

train_data_size = train_dataset.get_dataset_size()
print("train dataset length is:", train_data_size)

net = UNet3Plus(in_channels=1, n_classes=3)

# 预训练参数
if pretrain:
    param_dict = load_checkpoint(pretrain_path)
    load_param_into_net(net, param_dict)
    print('Load Successfully!')

loss = Tumour_Dice_CrossEntropy_Loss()
lr = Tensor(dynamic_lr(lr, warmup_ratio, train_data_size, num_epochs), mstype.float32)
optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, loss_scale=1024.0)

# 封装
net_with_loss = nn.WithLossCell(net, loss)
train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
train_net.set_train()

with open(log_file, 'w') as out:
    for epoch in range(num_epochs):
        step = 0
        print("============== Starting Training ==============")
        for d in train_dataset.create_dict_iterator():
            # print(np.sum(d['label'][:, 0]==1), np.sum(d['label'][:,1]==1), np.sum(d['label'][:,2]==1))
            # print(d['img'].shape, d['label'].shape)

            l_dice, l_ce, l = train_net(d["img"], d["label"])
            if step % 50 == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{time}, "
                      f"Epoch: [{epoch} / {num_epochs}], "
                      f"step: [{step} / {train_data_size}], "
                      f"loss_dice: {l_dice}, "
                      f"loss_ce: {l_ce}, "
                      f"loss: {l}, "
                      f"lr: {lr[epoch*train_data_size+step]}")
                print(f"{time}, "
                      f"Epoch: [{epoch} / {num_epochs}], "
                      f"step: [{step} / {train_data_size}], "
                      f"loss_dice: {l_dice}, "
                      f"loss_ce: {l_ce}, "
                      f"loss: {l}, "
                      f"lr: {lr[epoch*train_data_size+step]}", file=out)                
            step = step + 1
        print("============== End Training ==============")

        print("============== Starting Evaluation ==============")
        total_dice = evaluation(net, num_classes)   
        # https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/train_eval.html

        print("============== End Evaluation ==============")


        if total_dice[2] > best_tumour_dice:
            best_tumour_dice = total_dice[2]

            ms.save_checkpoint(net, 
                               os.path.join(ckpt_save_dir, 'best_tumour.ckpt'))
            print('checkpoint is saved successfully!')
print('Best Target Dice is: {}'.format(best_tumour_dice))