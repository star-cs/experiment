import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset import FullDataset, TestDataset
from SAM2UNet import SAM2UNet
from config import path_config, config_base
from torch.utils.tensorboard import SummaryWriter


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
    Sen = TP / (TP + FN)
    Spe = TN / (TN + FP)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    return Dice, IoU, Sen, Spe, Acc


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

@torch.no_grad
def test_medics(model, device, writer):
    metrics = Metrics(['Dice', 'IoU', 'Sen', 'Spe', 'Acc'])

    loss, wbce, wiou = 0, 0, 0
    test_dataloader = tqdm(test_dataloader)
    for batch in range(test_dataloader):
        x = batch['image']
        target = batch['label']
        x = x.to(device)
        target = target.to(device)
        pred0, pred1, pred2 = model(x)
                
        _Dice, _IoU, _Sen, _Spe, _Acc = evaluate(pred0, target)
                
        metrics.update(Dice = _Dice, IoU = _IoU, Sen = _Sen, 
                Spe = _Spe, Acc = _Acc)

        loss0, wbce0, wiou0 = structure_loss(pred0, target)
        loss1, wbce1, wiou1 = structure_loss(pred1, target)
        loss2, wbce2, wiou2 = structure_loss(pred2, target)
        loss = loss0 + loss1 + loss2
        wbce = wbce0 + wbce1 + wbce2
        wiou = wiou0 + wiou1 + wiou2
        print("Test loss:{} wbce:{} wiou:{}".format(
                                                        loss.item(),
                                                        wbce.item(), 
                                                        wiou.item()))
        writer.add_scalar('info/test_loss', loss.item(), epoch+1)
        writer.add_scalar('info/test_wbce', wbce.item(), epoch+1)
        writer.add_scalar('info/test_wiou', wiou.item(), epoch+1)

        metrics_result = metrics.mean(test_dataloader.size)
        print("Test Metrics Result:")
        print('Dice:  %.4f\nIoU: %.4f\nSen: %.4f\nSpe: %.4f\nAcc: %.4f, '
                    % (metrics_result['Dice'], metrics_result['IoU'], metrics_result['Sen'],
                    metrics_result['Spe'], metrics_result['Acc']))
        writer.add_scalar('info/metrics/Dice', metrics_result['Dice'], epoch+1)
        writer.add_scalar('info/metrics/IoU', metrics_result['IoU'], epoch+1)
        writer.add_scalar('info/metrics/Sen', metrics_result['Sen'], epoch+1)
        writer.add_scalar('info/metrics/Spe', metrics_result['Spe'], epoch+1)
        writer.add_scalar('info/metrics/Acc', metrics_result['Acc'], epoch+1)           

def main():  
    print(config_base)  
    print(path_config)
    
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
    os.makedirs(path_config['save_path'], exist_ok=True)
    writer = SummaryWriter(log_dir=path_config['tensorboard_path'])
    
    for epoch in range(config_base['epoch']):
        loss, wbce, wiou = 0, 0, 0
        dataloader = tqdm(dataloader)
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0, wbce0, wiou0 = structure_loss(pred0, target)
            loss1, wbce1, wiou1 = structure_loss(pred1, target)
            loss2, wbce2, wiou2 = structure_loss(pred2, target)
            loss += loss0 + loss1 + loss2
            wbce += wbce0 + wbce1 + wbce2
            wiou += wiou0 + wiou1 + wiou2
            
            loss.backward()
            optim.step()
           
        print("Train epoch:{}: loss:{} wbce:{} wiou:{}".format(epoch + 1,
                                                        loss.item(),
                                                        wbce.item(), 
                                                        wiou.item()))
        writer.add_scalar('info/loss', loss.item(), epoch+1)
        writer.add_scalar('info/wbce', wbce.item(), epoch+1)
        writer.add_scalar('info/wiou', wiou.item(), epoch+1)
                      
        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == config_base['epoch']:
            torch.save(model.state_dict(), os.path.join(path_config['save_path'], 'SAM2-UNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(path_config['save_path'], 'SAM2-UNet-%d.pth'% (epoch + 1)))

        test_medics(model, device, writer)

            

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