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

from dataset_Synapse import SynapseDataset

def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.Tensor([1])

    Dice = 2 * TP / (2 * TP + FP + FN)
    IoU = TP / (TP + FP + FN)
    Pre = TP/ (TP + FP)
    Recall = TP / (TP + FN)
    # Sen = TP / (TP + FN)
    # Spe = TN / (TN + FP)
    # Acc = (TP + TN) / (TP + FP + TN + FN)
    return Dice, IoU, Pre, Recall


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

def _one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(config_base['num_classes']):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def structure_loss(pred, mask):
    
    mask = _one_hot_encoder(mask.squeeze(1).long())
    
    # 计算权重张量 weit
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    # 二元交叉熵损失
    pred = torch.sigmoid(pred)
    wbce = torch.nn.BCELoss(reduction='none')(pred, mask)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
    # 计算交集和并集
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    
    # 计算 Dice 损失
    smooth = 1e-5
    dice = (2 * inter + smooth) / (union + smooth)
    dice_loss = 1 - dice
    
    # 返回加权的二元交叉熵损失和 Dice 损失的平均值，以及单独的二元交叉熵损失和 Dice 损失的平均值
    return (wbce + dice_loss).mean(), wbce.mean(), dice_loss.mean()


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
def test_medics(model, device, writer, test_dataloader, epoch):
    # metrics = Metrics(['Dice', 'IoU', 'Sen', 'Spe', 'Acc'])
    metrics = Metrics(['Dice', 'IoU', 'Pre', 'Recall']) 
    
    Loss, Wbce, Wdice = 0, 0, 0
    test_dataloader = tqdm(test_dataloader)
    for i, batch in enumerate(test_dataloader):
        x = batch['image']
        target = batch['label']
        x = x.to(device)
        target = target.to(device)
        pred0, pred1, pred2, pred3 = model(x)
                        
        _Dice, _IoU, _Pre, _Recall = evaluate(pred0, target)
                        
        # metrics.update(Dice = _Dice, IoU = _IoU, Sen = _Sen, 
        #                 Spe = _Spe, Acc = _Acc)
        metrics.update(Dice = _Dice, IoU = _IoU , Pre = _Pre, Recall = _Recall)
        loss0, wbce0, wdice0 = structure_loss(pred0, target)
        loss1, wbce1, wdice1 = structure_loss(pred1, target)
        loss2, wbce2, wdice2 = structure_loss(pred2, target)
        loss3, wbce3, wdice3 = structure_loss(pred3, target)
        Loss = loss0 + loss1 + loss2 + loss3
        Wbce = wbce0 + wbce1 + wbce2 + wbce3
        Wdice = wdice0 + wdice1 + wdice2 + wdice3

    print("Test epoch:{} loss:{} wbce:{} wdice:{}".format(epoch+1, 
                                                        Loss.item(),
                                                        Wbce.item(), 
                                                        Wdice.item()))
    writer.add_scalar('info/test_loss', Loss.item(), epoch+1)
    writer.add_scalar('info/test_wbce', Wbce.item(), epoch+1)
    writer.add_scalar('info/test_wdice', Wdice.item(), epoch+1)

    metrics_result = metrics.mean(len(test_dataloader))
    print("Test Metrics Result:")
    print('Dice: %.4f\nIoU: %.4f\nPre: %.4f\nRecall: %.4f' %(metrics_result['Dice'], metrics_result['IoU'],
                                                               metrics_result['Pre'], metrics_result['Recall']))
    writer.add_scalar('info/metrics/Dice', metrics_result['Dice'], epoch+1)
    writer.add_scalar('info/metrics/IoU', metrics_result['IoU'], epoch+1)
    writer.add_scalar('info/metrics/Pre', metrics_result['Pre'], epoch+1)
    writer.add_scalar('info/metrics/Recall', metrics_result['Recall'], epoch+1)
    return Loss.item(), Wbce.item(), Wdice.item(), metrics_result['Dice'],  metrics_result['IoU'], metrics_result['Pre'] , metrics_result['Recall']

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
    csv_writer.writerow(['time', 'step', 'train Loss', 'train wbce', 'train wdice',
                         'test Loss', 'test wbce', 'test wdice',
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
    dataset = SynapseDataset(data_dir=path_config['Synapse_image_path'],
                        mask_dir=path_config['Synapse_masks_path'],
                        size=config_base['image_size'],
                        mode='train',
                        list_dir=path_config['Synapse_lists_path'])
     
    test_loader = SynapseDataset(data_dir=path_config['Synapse_image_path'],
                        mask_dir=path_config['Synapse_masks_path'],
                        size=config_base['image_size'],
                        mode='test',
                        list_dir=path_config['Synapse_lists_path'])
    
    dataloader = DataLoader(dataset, 
                            batch_size=config_base['batch_size'], 
                            shuffle=True, num_workers=8)
    
    test_dataloader = DataLoader(test_loader, batch_size=1, shuffle=False)

    device = torch.device("cuda")
    model = SAM2UNet(path_config['hiera_path'], config_base['adapter_type'])
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": config_base['lr']}],
                       lr=config_base['lr'], weight_decay=config_base['weight_decay'])
    scheduler = CosineAnnealingLR(optim, config_base['epoch'], eta_min=1.0e-7)
    
    Min_Loss = 100000
    
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
        
        test_Loss, test_wbce, test_wdice, Dice, IoU, Pre, Recall = test_medics(model, device, writer, test_dataloader, epoch)

        if(Min_Loss > test_Loss):
            print('[Saving Basted Snapshot:]', os.path.join(model_save_path, 'SAM2-UNet-Best.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'SAM2-UNet-Best.pth'))
            Min_Loss = test_Loss
            
        elif(epoch+1 == config_base['epoch']):
            print('[Saving Lasted Snapshot:]', os.path.join(model_save_path, 'SAM2-UNet-Last.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'SAM2-UNet-Last.pth'))
            
        print()
        csv_writer.writerow((time.asctime(), epoch+1, Loss.item(), Wbce.item(), Wdice.item(), 
                      test_Loss, test_wbce, test_wdice, Dice, IoU, Pre,Recall))      
    
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