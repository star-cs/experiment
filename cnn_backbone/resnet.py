import torch
import torch.nn as nn
import torchvision.models as models

from config import config_base

class ResNet(nn.Module):
    
    def __init__(self, type:int = 50, is_pretrained:bool = True):
        super(ResNet, self).__init__()
        
        original_model = None
        self.channel_list = None
        if is_pretrained:
            if type == 50:
                original_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.channel_list = [256, 512, 1024, 2048]
            elif type == 34:
                original_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
                self.channel_list = [64, 128, 256, 512]
        else:
            if type == 50:
                original_model = models.resnet50(weights=None)
                self.channel_list = [256, 512, 1024, 2048]
                
                if(config_base['image_size'] == 256):
                    self.feature_size = [64, 32, 16, 8]
                elif(config_base['image_size'] == 512):
                    self.feature_size = [128, 64, 32, 16]

            elif type == 34:
                original_model = models.resnet34(weights=None)
                self.channel_list = [64, 128, 256, 512]
                
        
        self.stage0 = torch.nn.Sequential(*list(original_model.children())[:4]) 
        self.stage1 = original_model.layer1
        self.stage2 = original_model.layer2
        self.stage3 = original_model.layer3
        self.stage4 = original_model.layer4


    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        f1 = x 
        x = self.stage2(x)
        f2 = x 
        x = self.stage3(x)
        f3 = x 
        x = self.stage4(x)
        f4 = x 
        return [f1, f2, f3, f4]


if __name__ == "__main__":
    model = ResNet(34, is_pretrained = False)
    model.eval()
    x = torch.randn(1,3,512,512)
    y = model(x)
    #torch.save(model.state_dict(), 'cat.pth')
    for i in y:
        print(i.shape)