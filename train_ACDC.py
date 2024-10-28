import os
import argparse
import random
import time
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset import FullDataset, TestDataset
from SAM2UNet import SAM2UNet
from config import path_config, config_base, config_neck, config_decoder
from tensorboardX import SummaryWriter
import pandas as pd
import csv

from dataset_ACDC import ACDCdataset
import torch

from datasets.Synapse.utils import val_single_volume

def _one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(config_base['num_classes']):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def evaluate(pred, gt, num_classes):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    # 将标签转换为 one-hot 编码
    gt = _one_hot_encoder(gt.long())     # (batch_size, num_classes, height, width)

    # 初始化统计变量
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    TN = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)

    for c in range(num_classes):
        pred_binary = pred[:, c, :, :]
        pred_binary_inverse = (pred_binary == 0).float()

        gt_binary = gt[:, c, :, :]
        gt_binary_inverse = (gt_binary == 0).float()

        TP[c] = pred_binary.mul(gt_binary).sum().item()
        FP[c] = pred_binary.mul(gt_binary_inverse).sum().item()
        TN[c] = pred_binary_inverse.mul(gt_binary_inverse).sum().item()
        FN[c] = pred_binary_inverse.mul(gt_binary).sum().item()

    smooth = 1e-7

    # 计算指标
    Dice = 2 * TP / (2 * TP + FP + FN + smooth)
    IoU = TP / (TP + FP + FN + smooth)
    Pre = TP / (TP + FP + smooth)
    Recall = TP / (TP + FN + smooth)

    # 处理可能的除零错误
    Dice[torch.isnan(Dice)] = 0
    IoU[torch.isnan(IoU)] = 0
    Pre[torch.isnan(Pre)] = 0
    Recall[torch.isnan(Recall)] = 0

    return Dice.mean(), IoU.mean(), Pre.mean(), Recall.mean()

class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics


def structure_loss(pred, mask):
    
    mask = _one_hot_encoder(mask.long())     # (batch_size, num_classes, height, width)
    
    # 计算权重张量 weit
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    # 多分类交叉熵损失
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, mask.argmax(dim=1, keepdim=True).squeeze(1))
    ce_loss = (weit.sum(dim=1) * ce_loss).sum(dim=(1, 2)) / weit.sum(dim=(1, 2, 3))
    
    # 计算每个类别的 Dice 损失
    pred = F.softmax(pred, dim=1)  # 对预测结果应用 softmax
    inter = (pred * mask * weit).sum(dim=(2, 3))
    union = (pred * weit).sum(dim=(2, 3)) + (mask * weit).sum(dim=(2, 3))
    dice = (2. * inter + 1e-5) / (union + 1e-5)
    dice_loss = 1 - dice.mean(dim=1)
    
    # 返回加权的交叉熵损失和 Dice 损失的平均值，以及单独的交叉熵损失和 Dice 损失的平均值
    total_loss = (ce_loss + dice_loss).mean()
    ce_loss_mean = ce_loss.mean()
    dice_loss_mean = dice_loss.mean()
    
    return total_loss, ce_loss_mean, dice_loss_mean


def file_check(file_name):
    temp_file_name = file_name
    i = 1
    while i:
        #print(temp_file_name)
        #print(os.path.exists(temp_file_name))
        if os.path.exists(temp_file_name):
            name, suffix = file_name.split('.')
            name += '_' + str(i)
            temp_file_name = name+'.'+suffix
            i = i+1
        else:
            return temp_file_name

def dir_check(file_name):
    temp_file_name = file_name
    i = 1
    while i:
        #print(temp_file_name)
        #print(os.path.exists(temp_file_name))
        if os.path.exists(temp_file_name):
            name, suffix = os.path.split(file_name)
            suffix += '_' + str(i)
            temp_file_name = name +'/'+suffix
            i = i+1
        else:
            return temp_file_name

@torch.no_grad
def test_medics(model, device, writer, epoch):

    test_dataset = ACDCdataset(base_dir=path_config['ACDC_valid_path'],
                        list_dir=path_config['ACDC_lists_path'],
                        split='valid',
                        size=config_base['image_size'])
    
    test_dataloader = DataLoader(test_dataset, 
                            batch_size=1, 
                            shuffle=True, num_workers=8) 

    metrics = Metrics(['Dice', 'IoU', 'Pre', 'Recall']) 
    
    Loss, Wbce, Wiou = 0, 0, 0
    test_dataloader = tqdm(test_dataloader)
    for i, batch in enumerate(test_dataloader):
        x = batch['image']
        target = batch['label']
        x = x.to(device)
        target = target.to(device)
        pred0, pred1, pred2, pred3 = model(x)
                        
        _Dice, _IoU, _Pre, _Recall = evaluate(pred0, target, num_classes=config_base['num_classes'])        # target[bs,w,h]
        
        metrics.update(Dice = _Dice, IoU = _IoU , Pre = _Pre, Recall = _Recall)
        loss0, wbce0, wiou0 = structure_loss(pred0, target)         # target[bs,w,h]
        loss1, wbce1, wiou1 = structure_loss(pred1, target)
        loss2, wbce2, wiou2 = structure_loss(pred2, target)
        loss3, wbce3, wiou3 = structure_loss(pred3, target)
        Loss = loss0 + loss1 + loss2 + loss3
        Wbce = wbce0 + wbce1 + wbce2 + wbce3
        Wiou = wiou0 + wiou1 + wiou2 + wiou3

    print("Test epoch:{} loss:{} wbce:{} wiou:{}".format(epoch+1, 
                                                        Loss.item(),
                                                        Wbce.item(), 
                                                        Wiou.item()))
    writer.add_scalar('info/test_loss', Loss.item(), epoch+1)
    writer.add_scalar('info/test_wbce', Wbce.item(), epoch+1)
    writer.add_scalar('info/test_wiou', Wiou.item(), epoch+1)

    metrics_result = metrics.mean(len(test_dataloader))
    print("Test Metrics Result:")
    print('Dice: %.4f\nIoU: %.4f\nPre: %.4f\nRecall: %.4f' %(metrics_result['Dice'], metrics_result['IoU'],
                                                               metrics_result['Pre'], metrics_result['Recall']))
    writer.add_scalar('info/metrics/Dice', metrics_result['Dice'], epoch+1)
    writer.add_scalar('info/metrics/IoU', metrics_result['IoU'], epoch+1)
    writer.add_scalar('info/metrics/Pre', metrics_result['Pre'], epoch+1)
    writer.add_scalar('info/metrics/Recall', metrics_result['Recall'], epoch+1)
    return Loss.item(), Wbce.item(), Wiou.item(), metrics_result['Dice'],  metrics_result['IoU'], metrics_result['Pre'] , metrics_result['Recall']


def main():  
    print(config_base)
    print(path_config)
    print(config_neck)
    print(config_decoder)
    print()
    file_path = os.path.join(path_config['csv_path'], path_config['train_version'] + '.csv')
    file_path = file_check(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print('save csv path:', file_path)
    csv_f = open(file_path, 'w' , encoding='utf-8')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(['time', 'step', 'train Loss', 'train wbce', 'train wiou',
                         'test Loss', 'test wbce', 'test wiou',
                         'test metrics Dice', 'test metrics IoU', 'test metrics Pre', 'test metrics Recall'])
    
    model_save_path = os.path.join(path_config['save_path'], path_config['train_version'])
    model_save_path = dir_check(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    print('model save path:', model_save_path)

    tensorboard_save_path = os.path.join(path_config['tensorboard_path'], 
                                                path_config['train_version'])
    print('save tensorboard path:', tensorboard_save_path)
    writer = SummaryWriter(log_dir=tensorboard_save_path)

    print()
    dataset = ACDCdataset(base_dir=path_config['ACDC_train_path'],
                        list_dir=path_config['ACDC_lists_path'],
                        split='train',
                        size=config_base['image_size'])
    
    dataloader = DataLoader(dataset, 
                            batch_size=config_base['batch_size'], 
                            shuffle=True, num_workers=8)
    
    print("train dataset {}".format(len(dataset)))
    
    device = torch.device("cuda")
    model = SAM2UNet(path_config['hiera_path'], config_base['adapter_type'])
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": config_base['lr']}],
                       lr=config_base['lr'], weight_decay=config_base['weight_decay'])
    scheduler = CosineAnnealingLR(optim, config_base['epoch'], eta_min=1.0e-7)
    
    Max_Dice = 0
    
    for epoch in range(config_base['epoch']):
        Loss, Wbce, Wdice = 0, 0, 0
        dataloader = tqdm(dataloader)
        for i, batch in enumerate(dataloader):
        
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2, pred3 = model(x)
            loss0, wbce0, wdice0 = structure_loss(pred0, target)
            loss1, wbce1, wdice1 = structure_loss(pred1, target)
            loss2, wbce2, wdice2 = structure_loss(pred2, target)
            loss3, wbce3, wdice3 = structure_loss(pred3, target)
            loss = loss0 + loss1 + loss2 + loss3
            wbce = wbce0 + wbce1 + wbce2 + wbce3
            wdice = wdice0 + wdice1 + wdice2 + wdice3
            Loss += loss
            Wbce += wbce
            Wdice += wdice
            
            loss.backward()
            optim.step()
            
        print("Train epoch:{} loss:{} wbce:{} wdice:{}".format(epoch + 1,
                                                        Loss.item(),
                                                        Wbce.item(), 
                                                        Wdice.item()))
        writer.add_scalar('info/loss', Loss.item(), epoch+1)
        writer.add_scalar('info/wbce', Wbce.item(), epoch+1)
        writer.add_scalar('info/wdice', Wdice.item(), epoch+1)
                      
        scheduler.step()
        
        test_loss, test_wbce, test_wdice, Dice, IoU, Pre, Recall = test_medics(model, device, writer, epoch)
        
        writer.add_scalar('info/test_loss', test_loss, epoch+1)
        writer.add_scalar('info/test_wbce', test_wbce, epoch+1)
        writer.add_scalar('info/test_wdice', test_wdice, epoch+1)
        writer.add_scalar('info/test_dice', Dice, epoch+1)
        writer.add_scalar('info/test_iou', IoU, epoch+1)
        writer.add_scalar('info/test_pre', Pre, epoch+1)
        writer.add_scalar('info/test_Recell', Recall, epoch+1)   
        
        if(Max_Dice < Dice):
            print('[Saving Basted Snapshot:]', os.path.join(model_save_path, 'Best_' + str(epoch+1) + '_.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'Best_' + str(epoch+1) + '_.pth'))
            Max_Dice = Dice
            
        elif(epoch+1 == config_base['epoch']):
            print('[Saving Lasted Snapshot:]', os.path.join(model_save_path, 'Last_' + str(epoch+1) + '_.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'Last_' + str(epoch+1) + '_.pth'))
            
        print()
        
        csv_writer.writerow((time.asctime(), epoch+1, Loss.item(), Wbce.item(), Wdice.item(), 
                             test_loss, test_wbce, test_wdice, Dice, IoU, Pre,Recall))      
    
    writer.close()
    
# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main()