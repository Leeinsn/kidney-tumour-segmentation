from model.dataloader import kidney
from model.transforms_3d import Normalize, LabelEncode, ExpandChannelWithLabel

from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms.transforms import Compose
from mindspore import Model
from tqdm import tqdm
import numpy as np
import math


def calculate_iou(y_pred, y, num_classes):
    """
    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        label: ground truth, the first dim is batch.
    """
    y_pred = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
    intersect = y_pred[y_pred == y]
    
    area_intersect, _ = np.histogram(intersect, bins=num_classes, range=(0, num_classes))
    area_pred_label, _ = np.histogram(y_pred, bins=num_classes, range=(0, num_classes))
    area_label, _ = np.histogram(y, bins=num_classes, range=(0, num_classes))
    # area_union = area_pred_label + area_label - area_intersect
    # print(area_intersect, area_pred_label, area_label)
    
    return area_intersect, area_pred_label, area_label


def calculate_dice(total_area_intersect, total_area_pred_label, total_area_label):
    dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
    # acc = area_intersect / area_label
    for idx, ele in enumerate(dice):
        if math.isnan(ele):
            dice[idx] = 0
    return dice


def evaluation(net, num_classes, out):
    eval_dataset = kidney(isTrain=False)
    eval_dataset = GeneratorDataset(source=eval_dataset, column_names=['img', 'label'], shuffle=False)
    trans = Compose([Normalize(mean=119.84, std=103.80),
                     ExpandChannelWithLabel(),
                ])
    eval_dataset = eval_dataset.map(operations=trans, input_columns=["img", "label"])
    eval_dataset = eval_dataset.batch(1)
    
    eval_data_size = eval_dataset.get_dataset_size()
    print("val dataset length is:", eval_data_size)
    
    total_area_intersect = np.zeros(num_classes)
    total_area_pred_label = np.zeros(num_classes)
    total_area_label = np.zeros(num_classes)
    
    # 构建评估网络
    eval_net = Model(net)
    
    for batch in tqdm(eval_dataset.create_dict_iterator(num_epochs=1), total=eval_data_size):
        image = batch["img"]  # 1*1*256*256
        label = batch["label"]  # 1*1*256*256
        # print("current image shape is {}".format(image.shape), flush=True)
        pred = eval_net.predict(image)
        pred = pred.asnumpy()
        label = label.asnumpy()
        
        area_intersect, area_pred_label, area_label = calculate_iou(pred, label, num_classes=2)
        total_area_intersect += area_intersect
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    
    dice = calculate_dice(total_area_intersect, total_area_pred_label, total_area_label)
    print(dice)
    print("mean Dice on case 200 ~ 209 is")
    
    print("    background: {:.2f}, kidney: {:.2f} ".format(dice[0]*100, dice[1]*100))
    print("    background: {:.2f}, kidney: {:.2f} ".format(dice[0]*100, dice[1]*100), file=out)
    return dice[0]*100, dice[1]*100