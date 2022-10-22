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
log_file = os.path.join(ckpt_save_dir, 'log.txt')

if not os.path.exists(ckpt_save_dir):
    os.makedirs(ckpt_save_dir)
# ckpt_path = './work_dirs/checkpoint/ckpt_20221008/best_kidney.ckpt'
# seg_content = 'kidney'

train_dataset = kidney()
train_dataset = GeneratorDataset(source=train_dataset, column_names=['img', 'label'], shuffle=True)

trans = Compose([Normalize(mean=119.84, std=103.80),
                 RandomVerticalFlip(),
                 RandomVerticalFlip(),
                 LabelEncode(num_classes),
                 ExpandChannel(),
                 # RandomRotFlip(),
                 # AjustContrast(),
                 # GaussianNoise(),
                ])
train_dataset = train_dataset.map(operations=trans, input_columns=["img", "label"], num_parallel_workers=1)

train_dataset = train_dataset.batch(batch_size)

train_data_size = train_dataset.get_dataset_size()
print("train dataset length is:", train_data_size)

net = UNet3d_(num_classes)

loss = Dice_CrossEntropy_Loss()

lr = Tensor(dynamic_lr(lr, warmup_ratio, train_data_size, num_epochs), mstype.float32)
optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, loss_scale=1024.0)


# 预训练参数
if pretrain:
    param_dict = load_checkpoint(pretrain_path)
    load_param_into_net(net, param_dict)
    print('Load Successfully!')

net_with_loss = nn.WithLossCell(net, loss)
    
# 定义训练网络，封装网络和优化器
train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
train_net.set_train()

# 真正训练迭代过程
with open(log_file, 'w') as out:
    for epoch in range(num_epochs):
        step = 0
        print("============== Starting Training ==============")
        for d in train_dataset.create_dict_iterator():
            # print(d['img'].shape, d['label'].shape)
            l = train_net(d["img"], d["label"])
            if step % 50 == 0:
                    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{time}, "
                          f"Epoch: [{epoch} / {num_epochs}], "
                          f"step: [{step} / {train_data_size}], "
                          f"loss: {l}, "
                          f"lr: {lr[epoch*train_data_size+step]}")
                    print(f"{time}, "
                          f"Epoch: [{epoch} / {num_epochs}], "
                          f"step: [{step} / {train_data_size}], "
                          f"loss: {l}, "
                          f"lr: {lr[epoch*train_data_size+step]}", file=out)                
            step = step + 1
        print("============== End Training ==============")
        print("============== Starting Evaluation ==============")
        total_dice = evaluation(net, num_classes, out)   # https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/train_eval.html

        print("============== End Evaluation ==============")
        if total_dice[1] > best_dice:
            best_dice = total_dice[1]
            
            ms.save_checkpoint(net, 
                               os.path.join(ckpt_save_dir, 'best_kidney.ckpt'))
            print('checkpoint is saved successfully!')

print('Best Target Dice is: {}'.format(best_dice))