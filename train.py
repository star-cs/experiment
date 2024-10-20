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
from common import bce_loss, dice_coeff
from dataset import FullDataset, TestDataset
from SAM2UNet import SAM2UNet
from config import path_config, config_base, config_neck, config_decoder, config_prompt_encoder, config_loss
from tensorboardX import SummaryWriter
import pandas as pd
import csv

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

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # 二元交叉熵损失
    pred = torch.sigmoid(pred)
    wbce = torch.nn.BCELoss(reduction='none')(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    
    wiou = 1 - (inter + 1)/(union - inter+ 1)
    return (wbce + wiou).mean(), wbce.mean(), wiou.mean()

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
    
    Loss, Loss_ce, Loss_dice = 0, 0, 0
    test_dataloader = tqdm(test_dataloader)
    for i, batch in enumerate(test_dataloader):
        x = batch['image']
        target = batch['label']
        x = x.to(device)
        target = target.to(device)
        outputs = model(x)
        loss, loss_ce, loss_dice = calc_riga_loss(outputs, target, config_loss['dice_param'])
        Loss += loss
        Loss_ce += loss_ce
        Loss_dice += loss_dice
        
        seg_output = torch.sigmoid(outputs['masks'])
        
        _Dice, _IoU, _Pre, _Recall = evaluate(seg_output, target)
                        
        metrics.update(Dice = _Dice, IoU = _IoU, Pre = _Pre, Recall = _Recall)

    print("Test epoch:{} loss:{} Loss_ce:{} Loss_dice:{}".format(epoch+1, 
                                                        Loss.item(),
                                                        Loss_ce.item(), 
                                                        Loss_dice.item()))
    writer.add_scalar('info/test_loss', Loss.item(), epoch+1)
    writer.add_scalar('info/test_Loss_ce', Loss_ce.item(), epoch+1)
    writer.add_scalar('info/test_Loss_dice', Loss_dice.item(), epoch+1)

    metrics_result = metrics.mean(len(test_dataloader))
    print("Test Metrics Result:")
    print('Dice: %.4f\nIoU: %.4f\nPre: %.4f\nRecall: %.4f' %(metrics_result['Dice'], metrics_result['IoU'],
                                                               metrics_result['Pre'], metrics_result['Recall']))
    writer.add_scalar('info/metrics/Dice', metrics_result['Dice'], epoch+1)
    writer.add_scalar('info/metrics/IoU', metrics_result['IoU'], epoch+1)
    writer.add_scalar('info/metrics/Pre', metrics_result['Pre'], epoch+1)
    writer.add_scalar('info/metrics/Recall', metrics_result['Recall'], epoch+1)
    return Loss.item(), Loss_ce.item(), Loss_dice.item(), metrics_result['Dice'],  metrics_result['IoU'], metrics_result['Pre'] , metrics_result['Recall']


def calc_riga_loss(outputs, label_batch, dice_weight:float=0.8):
    logits0 = outputs['masks'][:, 0]
    pred0 = torch.nn.Sigmoid()(logits0)
    label_batch0 = (label_batch[:, 0] > 0) * 1.0
    loss_ce0 = bce_loss(pred=pred0, label=label_batch0)
    loss_dice0 = dice_coeff(pred=pred0, label=label_batch0)
    loss0 = (1 - dice_weight) * loss_ce0 + dice_weight * loss_dice0

    logits1 = outputs['masks'][:, 1]
    pred1 = torch.nn.Sigmoid()(logits1)
    label_batch1 = (label_batch[:, 0] == 2) * 1.0
    loss_ce1 = bce_loss(pred=pred1, label=label_batch1)
    loss_dice1 = dice_coeff(pred=pred1, label=label_batch1)
    loss1 = (1 - dice_weight) * loss_ce1 + dice_weight * loss_dice1

    return loss0+loss1, loss_ce0+loss_ce1, loss_dice0+loss_dice1


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
    csv_writer.writerow(['time', 'step', 'train Loss', 'train Loss_ce', 'train Loss_dice',
                         'test Loss', 'test Loss_ce', 'test Loss_dice',
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
    dataset = FullDataset(path_config['train_image_path'],
                        path_config['train_mask_path'],
                        config_base['image_size'],
                        mode='train')
     
    test_loader = FullDataset(path_config['test_image_path'],
                        path_config['test_mask_path'],
                        config_base['image_size'],
                        mode='test')
    
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
        Loss, Loss_ce, Loss_dice = 0, 0, 0
        dataloader = tqdm(dataloader)
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            
            outputs = model(x)
            loss, loss_ce, loss_dice = calc_riga_loss(outputs, target, config_loss['dice_param'])
            Loss += loss    
            Loss_ce += loss_ce
            Loss_dice += loss_dice
                  
            loss.backward()
            optim.step()
           
        print("Train epoch:{} loss:{} Loss_ce:{} Loss_dice:{}".format(epoch + 1,
                                                        Loss.item(),
                                                        Loss_ce.item(), 
                                                        Loss_dice.item()))
        writer.add_scalar('info/loss', Loss.item(), epoch+1)
        writer.add_scalar('info/loss_ce', Loss_ce.item(), epoch+1)
        writer.add_scalar('info/loss_dice', Loss_dice.item(), epoch+1)
                      
        scheduler.step()
        
        test_Loss, test_Loss_ce, test_Loss_dice, Dice, IoU, Pre, Recall = test_medics(model, device, writer, test_dataloader, epoch)

        if(Min_Loss > test_Loss):
            print('[Saving Basted Snapshot:]', os.path.join(model_save_path, 'SAM2-UNet-Best.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'SAM2-UNet-Best.pth'))
            Min_Loss = test_Loss
            
        elif(epoch+1 == config_base['epoch']):
            print('[Saving Lasted Snapshot:]', os.path.join(model_save_path, 'SAM2-UNet-Last.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'SAM2-UNet-Last.pth'))
            
        print()
        csv_writer.writerow((time.asctime(), epoch+1, Loss.item(), Loss_ce.item(), Loss_dice.item(), 
                      test_Loss, test_Loss_ce, test_Loss_dice, Dice, IoU, Pre, Recall))      
    
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