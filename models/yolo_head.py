import torch
import torch.nn as nn
from models.hrnet import get_Hrnet,BN_MOMENTUM
from models.darknet import darknet53
import time
from utils.decoder import yoloDecode
from collections import OrderedDict
import torch.nn.functional as F
import math
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Dark_fuse(nn.Module):
    def __init__(self,config):
        super(Dark_fuse, self).__init__()
        cfg=config.MODEL
        self.channels = cfg['STAGE4']['NUM_CHANNELS'][1:]
        self.layer3 = self._make_layer(2)  #1024
        self.layer2 = self._make_layer(1)  # 512
        self.layer1 = self._make_layer(0)  # 256

        self.de_channel2= self.de_channel(512)
        self.de_channel1= self.de_channel(256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def de_channel(self,channl):
        return nn.Sequential(
            nn.Conv2d(channl, channl // 2, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channl // 2),
            nn.LeakyReLU(0.1))

    def _make_layer(self,id):
        layers = []
        for i in range(3):
            if i ==0 and id < 2:
                channl=self.channels[id] + self.channels[id]//2
            else:
                channl=self.channels[id]
            layers.append(('yolo-%d-conv0%d'%(id,i),nn.Conv2d(channl,self.channels[id]//2, kernel_size=1,
                               stride=1, padding=0, bias=False)))
            layers.append(('yolo-%d-bn0%d' % (id, i), nn.BatchNorm2d(self.channels[id]//2)))
            layers.append(('yolo-%d-relu0%d' % (id, i),nn.LeakyReLU(0.1)))
            if i <  2 :
                layers.append(('yolo-%d-conv1%d'%(id,i),nn.Conv2d(self.channels[id]//2,self.channels[id], kernel_size=3,
                                   stride=1, padding=1, bias=False)))
                layers.append(('yolo-%d-bn1%d' % (id, i), nn.BatchNorm2d(self.channels[id])))
                layers.append(('yolo-%d-relu1%d' % (id, i),nn.LeakyReLU(0.1)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, input):
        out13=self.layer3(input[2])

        conv26=F.interpolate(self.de_channel2(out13), scale_factor=2,mode='nearest')
        conv26=torch.cat([input[1],conv26],dim=1)
        out26 = self.layer2(conv26)

        conv52=F.interpolate(self.de_channel1(out26), scale_factor=2,mode='nearest')
        conv52=torch.cat([input[0],conv52],dim=1)
        out52 = self.layer1(conv52)
        return out52,out26,out13

class Yolodet(nn.Module):
    def __init__(self,config,pretrained=True):
        super(Yolodet,self).__init__()
        cfg=config.MODEL
        self.heads=cfg.HEADS

        if cfg.BACKBONE is 'darknet':
            self.base=darknet53(cfg,is_train=pretrained)
            self.feat_fuse=Dark_fuse(config)
            channel_up = 0.5
        elif  cfg.BACKBONE is 'hrnet':
            self.base = get_Hrnet(cfg, is_train=pretrained)
            self.feat_fuse=None
            channel_up = 1
        else:
            raise Exception("not support")

        channels = cfg['STAGE4']['NUM_CHANNELS'][1:]
        for head in self.heads:
            id=int(head.split('-')[-1])
            classes = self.heads[head]
            in_channel=int(channels[id]*channel_up)
            fc = nn.Sequential(
                nn.Conv2d(in_channel, in_channel*2,
                          kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(in_channel*2,momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel*2, classes,
                          kernel_size=1, stride=1,
                          padding=0, bias=True))
            fill_fc_weights(fc)
            self.__setattr__(head, fc)
            decode=yoloDecode(id,config)
            self.__setattr__(head+'-decode', decode)

    def forward(self, x):
        x = self.base(x)[1:]
        if self.feat_fuse is not None:
            x=self.feat_fuse(x)
        ret = {}
        for head in self.heads:
            id = int(head.split('-')[-1])
            ret[head] = self.__getattr__(head)(x[id])
            ret[head + '-decode']=self.__getattr__(head + '-decode')(ret[head])
        return ret

if __name__ == '__main__':
    import hrnetyolo_config
    import darkyolo_config
    input = torch.rand([3, 3, 416, 416])
    model=Yolodet(darkyolo_config)
    out = model(input)
    time.sleep(3)
