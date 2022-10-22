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

from mindspore import nn
import mindspore as ms
from mindspore import ops
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.nn.loss.loss import LossBase
from model.model_utils.config import config

class SoftmaxCrossEntropyWithLogits(LossBase):
    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        self.cast = P.Cast()
        self.reduce_mean = P.ReduceMean()
        self.num_classes = config.num_classes

    def construct(self, logits, label):
        logits = self.transpose(logits, (0, 2, 3, 4, 1))  # B, C, D, H, W -> B, D, H, W, C
        label = self.transpose(label, (0, 2, 3, 4, 1))
        label = self.cast(label, mstype.float32)  # 转换输入类型
        loss = self.reduce_mean(self.loss_fn(self.reshape(logits, (-1, self.num_classes)), \
                                self.reshape(label, (-1, self.num_classes))))
        return self.get_loss(loss)


class Dice_CrossEntropy_Loss(LossBase):
    def __init__(self):
        super(Dice_CrossEntropy_Loss, self).__init__()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.reduce_mean = P.ReduceMean()
        self.softmax = P.Softmax(axis=4)  # 通道维度
        # self.ignore_background = True
        self.loss_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, logits, label):
        # B * N * D * H * W  (N: num_classes)
        logits = self.transpose(logits, (0, 2, 3, 4, 1))  # B, N, D, H, W -> B, D, H, W, N
        label = self.transpose(label, (0, 2, 3, 4, 1))
        label = self.cast(label, mstype.float32)  # 转换输入类型
        
        soft_logits = self.softmax(logits)
        # ignore background
        soft_logits = soft_logits[:, :, :, :, 1:]
        soft_label = label[:, :, :, :, 1:]
        
        intersect = (soft_logits * soft_label).sum()
        union = soft_logits.sum() + soft_label.sum()
        
        loss_dice = 1 - ((2.0 * intersect + 1) / (union + 1))
        
        logits = self.reshape(logits, (-1, 2))
        label = self.reshape(label, (-1, 2))
        loss_ce = self.reduce_mean(self.loss_ce(logits, label))

        return loss_ce + loss_dice

    
class Tumour_Dice_CrossEntropy_Loss(LossBase):
    def __init__(self):
        super(Tumour_Dice_CrossEntropy_Loss, self).__init__()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.reduce_mean = P.ReduceMean()
        self.softmax = P.Softmax(axis=-1)  # 通道维度
        self.loss_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    def construct(self, logits, label):  # 4*3*256*256
        # logits: b, c, h, w  label: b, c, h, w
        logits = self.transpose(logits, (0, 2, 3, 1))
        label = self.transpose(label, (0, 2, 3, 1))  # 4*256*256*3
        
        soft_logits = self.softmax(logits)
        
        # 计算kidney的dice
        # soft_logits_kidney = soft_logits[:, :, :, 1]
        # soft_label_kidney = label[:, :, :, 1]
        
        # intersect_kidney = (soft_logits_kidney * soft_label_kidney).sum()
        # union_kidney = soft_logits_kidney.sum() + soft_label_kidney.sum()
        
        # dice_kidney = (2.0 * intersect_kidney + 1) / (union_kidney + 1)
        
        # 计算tumour的dice
        soft_logits_tumour = soft_logits[:, :, :, 2]
        soft_label_tumour = label[:, :, :, 2]
        intersect_tumour = (soft_logits_tumour * soft_label_tumour).sum()
        union_tumour = soft_logits_tumour.sum() + soft_label_tumour.sum()
        
        dice_tumour = (2.0 * intersect_tumour) / union_tumour
        
        # loss_dice_kidney = ops.pow(-ops.log(dice_kidney), 0.3) * 0.4
        # loss_dice_tumour = ops.pow(-ops.log(dice_tumour), 0.3) * 0.6
        # loss_dice = loss_dice_kidney + loss_dice_tumour
        
        loss_dice = 1 - dice_tumour
        
        logits = self.reshape(logits, (-1, 3))
        label = self.reshape(label, (-1, 3))
        
        # loss_ce = 0.28 * loss_ce_bg + 0.28 * loss_ce_kidney + 0.44 * loss_ce_tumour
        loss_ce = self.loss_ce(logits, label) 
        
        return loss_dice, loss_ce, loss_dice + loss_ce
        
        
        