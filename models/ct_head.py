import torch
import torch.nn as nn
from models.hrnet import get_Hrnet,BN_MOMENTUM
import numpy as np
import torch.nn.functional as F
import time


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Ctdet(nn.Module):

    def __init__(self,config,pretrained=True):

        super(Ctdet,self).__init__()
        config=config.MODEL
        self.heads=config.HEADS
        self.base=get_Hrnet(config,is_train=pretrained)
        last_inp_channels = np.int(np.sum(config['STAGE4']['NUM_CHANNELS']))
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels,
                          kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(last_inp_channels,momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_inp_channels, classes,
                          kernel_size=1, stride=1,
                          padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')
        x = torch.cat([x[0], x1, x2, x3], 1)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret


if __name__ == '__main__':
    from config import MODEL

    input = torch.rand([3, 3, 960, 512], device='cuda')
    model=Ctdet(MODEL)
    model.cuda()
    out = model(input)

    time.sleep(3)
