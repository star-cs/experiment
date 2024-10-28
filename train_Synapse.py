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

from dataset_Synapse import RandomGenerator, SynapseDataset
from torchvision import transforms
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
    gt = _one_hot_encoder(gt.squeeze(1).long())     # (batch_size, num_classes, height, width)

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

    # 计算指标
    Dice = 2 * TP / (2 * TP + FP + FN)
    IoU = TP / (TP + FP + FN)
    Pre = TP / (TP + FP)
    Recall = TP / (TP + FN)

    # 处理可能的除零错误
    Dice[torch.isnan(Dice)] = 0
    IoU[torch.isnan(IoU)] = 0
    Pre[torch.isnan(Pre)] = 0
    Recall[torch.isnan(Recall)] = 0

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
def test_medics(model):
    db_test = SynapseDataset(base_dir=path_config['Synapse_test_path'],
                        list_dir=path_config['Synapse_lists_path'],
                        split='test_vol',
                        nclass=config_base['num_classes'])
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # print("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    testloader = tqdm(testloader)
    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0] # [1,148,512,512] [1,148,512,512] 
        metric_i = val_single_volume(image, label, model, classes=config_base['num_classes'], 
                                     patch_size=[config_base['image_size'], config_base['image_size']],
                                      case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    return performance, metric_list


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
                         'mean_Dice', 'Aorta', 'Gallbladder', 'Kidney(L)', 'Kidney(R)', 'Liver', 'Pancreas', 'Spleen', 'Stomach'])
   
    model_save_path = os.path.join(path_config['save_path'], path_config['train_version'])
    model_save_path = dir_check(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    print('model save path:', model_save_path)

    tensorboard_save_path = os.path.join(path_config['tensorboard_path'], 
                                                path_config['train_version'])
    print('save tensorboard path:', tensorboard_save_path)
    writer = SummaryWriter(log_dir=tensorboard_save_path)

    print()
    dataset = SynapseDataset(base_dir=path_config['Synapse_train_path'],
                        list_dir=path_config['Synapse_lists_path'],
                        split='train',
                        nclass=config_base['num_classes'],
                        transform=transforms.Compose(
                        [RandomGenerator(output_size=[config_base['image_size'], config_base['image_size']])]))
     
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
        
        performance, metric_list = test_medics(model)
        Aorta, Gallbladder, KidneyL, KidneyR, Liver, Pancreas, Spleen, Stomach = metric_list
        
        writer.add_scalar('info/dice', performance, epoch+1)
        writer.add_scalar('info/Aorta', Aorta, epoch+1)
        writer.add_scalar('info/Gallbladder', Gallbladder, epoch+1)
        writer.add_scalar('info/KidneyL', KidneyL, epoch+1)
        writer.add_scalar('info/KidneyR', KidneyR, epoch+1)
        writer.add_scalar('info/Liver', Liver, epoch+1)
        writer.add_scalar('info/Pancreas', Pancreas, epoch+1)
        writer.add_scalar('info/Spleen', Spleen, epoch+1)
        writer.add_scalar('info/Stomach', Stomach, epoch+1)
        print("Test Metrics Result:")
        print(' Dice: %.4f\n Aorta: %.4f\n Gallbladder: %.4f\n KidneyL: %.4f\n KidneyR: %.4f\n Liver: %.4f\n Pancreas: %.4f\n Spleen: %.4f\n Stomach: %.4f' % (performance, Aorta, Gallbladder, KidneyL, KidneyR, Liver, Pancreas, Spleen, Stomach))
        
        if(Max_Dice < performance):
            print('[Saving Basted Snapshot:]', os.path.join(model_save_path, 'Best_' + str(epoch+1) + '_.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'Best_' + str(epoch+1) + '_.pth'))
            Max_Dice = performance
            
        elif(epoch+1 == config_base['epoch']):
            print('[Saving Lasted Snapshot:]', os.path.join(model_save_path, 'Last_' + str(epoch+1) + '_.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'Last_' + str(epoch+1) + '_.pth'))
            
        print()
        
        csv_writer.writerow((time.asctime(), epoch+1, Loss.item(), Wbce.item(), Wdice.item(), 
                             performance, Aorta, Gallbladder, KidneyL, KidneyR, Liver, Pancreas, Spleen, Stomach))      
    
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