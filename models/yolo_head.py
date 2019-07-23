import torch
import torch.nn as nn
from models.backbone import darknet53,darknet21,hrnet
import time
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from utils.torch_utils import _sigmoid,exp,pow

class yoloDecode(nn.Module):
    def __init__(self,id,config):
        super(yoloDecode,self).__init__()
        self.scale = config.DATASET.DOWN_RATION[id]
        self.H=config.DATASET.INPUT_H //self.scale
        self.W=config.DATASET.INPUT_W // self.scale
        self.use_giou = config.DATASET.USE_GIOU
        anchors = np.array(config.DATASET.ANCHORS[id])*(np.array([config.DATASET.INPUT_W,config.DATASET.INPUT_H])[np.newaxis,:])
        self.anchors=torch.from_numpy(anchors).float()
        x = torch.arange(self.W)[np.newaxis, :].expand([self.W, self.H])
        y = torch.arange(self.H)[:, np.newaxis].expand([self.W, self.H])
        xy_grid = torch.cat([x[:, :, np.newaxis], y[:, :, np.newaxis]], dim=-1)[np.newaxis, :, :, np.newaxis, :]
        self.xy_grid=xy_grid.float()
        self.wh_activation = exp() if not self.use_giou else pow()


    def _apply(self, fn):
        self.xy_grid = fn(self.xy_grid)
        self.anchors = fn(self.anchors)

    def forward(self, pred):
        pred = pred.permute(0, 2, 3, 1).contiguous() #  B,H,W,C
        B,H,W,C=pred.size()
        assert H==self.H
        assert W==self.W
        pred = pred.view(B,H,W,3,C//3)
        pred_conv_dxdy = pred[:, :, :, :, 0:2]
        pred_conv_dwdh = pred[:, :, :, :, 2:4]
        pred_conv_conf = pred[:, :, :, :, 4:5]
        pred_conv_prob = pred[:, :, :, :, 5:]
        pred_xy = (_sigmoid(pred_conv_dxdy)+ self.xy_grid) * self.scale
        pred_wh = (self.wh_activation(pred_conv_dwdh) * self.anchors)
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = _sigmoid(pred_conv_conf)
        pred_prob = _sigmoid(pred_conv_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)
        return pred_bbox

class Dark_fuse(nn.Module):
    def __init__(self,config):
        super(Dark_fuse, self).__init__()
        cfg=config.NECK
        self.channels = cfg.IN_CHANNEL
        self.layer3 = self._make_layer(2)  #1024
        self.layer2 = self._make_layer(1)  # 512
        self.layer1 = self._make_layer(0)  # 256

        self.de_channel2= self.de_channel(512)
        self.de_channel1= self.de_channel(256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def de_channel(self,channl):
        return nn.Sequential(OrderedDict(
            [('conv',nn.Conv2d(channl, channl // 2, kernel_size=1,
                      stride=1, padding=0, bias=False)),
            ('bn',nn.BatchNorm2d(channl // 2)),
            ('relu',nn.LeakyReLU(0.1))]))

    def _make_layer(self,id):
        m = nn.ModuleList()
        for i in range(3):
            if i ==0 and id < 2:
                channl=self.channels[id] + self.channels[id]//2
            else:
                channl=self.channels[id]
            m.append(nn.Sequential(OrderedDict([('conv',nn.Conv2d(channl,self.channels[id]//2, kernel_size=1,
                               stride=1, padding=0, bias=False)),
                                ('bn' , nn.BatchNorm2d(self.channels[id]//2)),
                                ('relu',nn.LeakyReLU(0.1))])))
            if i < 2 :
                m.append(nn.Sequential(OrderedDict([('conv',nn.Conv2d(self.channels[id]//2,self.channels[id], kernel_size=3,
                                   stride=1, padding=1, bias=False)),
                                ('bn' , nn.BatchNorm2d(self.channels[id])),
                                ('relu',nn.LeakyReLU(0.1))])))
        return nn.Sequential(*m)

    def forward(self, input):
        out13=self.layer3(input[2])

        conv26=F.interpolate(self.de_channel2(out13), scale_factor=2,mode='nearest')
        conv26=torch.cat([input[1],conv26],dim=1)
        out26 = self.layer2(conv26)

        conv52=F.interpolate(self.de_channel1(out26), scale_factor=2,mode='nearest')
        conv52=torch.cat([input[0],conv52],dim=1)
        out52 = self.layer1(conv52)

        return out52,out26,out13


class Hrnet_fuse(nn.Module):
    def __init__(self):
        super(Hrnet_fuse, self).__init__()

    def forward(self, input):

        return input

class Yolodet(nn.Module):
    def __init__(self,config,pretrained=True):
        super(Yolodet,self).__init__()
        cfg=config.HEAD
        self.config = config
        self.heads=cfg.OUT_HEADS
        channels = cfg.IN_CHANNEL

        if cfg.BACKBONE is 'darknet':
            self.backbone=darknet53(config,is_train=pretrained)
            self.feat_fuse=Dark_fuse(config)
        elif cfg.BACKBONE is 'hrnet':
            self.backbone=hrnet(config,is_train=pretrained)
            self.feat_fuse=Hrnet_fuse()

        for head in self.heads:
            id=int(head.split('-')[-1])
            classes = self.heads[head]
            in_channel=int(channels[id])
            fc = nn.Sequential(OrderedDict(
                [('conv',nn.Conv2d(in_channel, in_channel*2,
                          kernel_size=3, padding=1, bias=False)),
                 ('bn',nn.BatchNorm2d(in_channel*2)),
                 ('relu',nn.LeakyReLU(0.1)),
                 ('conv_out',nn.Conv2d(in_channel*2, classes,
                          kernel_size=1, stride=1,
                          padding=0, bias=True))]))
            for m in fc.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            self.__setattr__(head, fc)
            decode=yoloDecode(id,config)
            self.__setattr__(head+'-decode', decode)

    def forward(self, x):
        x = self.backbone(x)
        if self.feat_fuse is not None:
            x=self.feat_fuse(x)
        ret = {}
        for head in self.heads:
            id = int(head.split('-')[-1])
            ret[head] = self.__getattr__(head)(x[id])
            ret[head + '-decode']=self.__getattr__(head + '-decode')(ret[head])
        return ret

    def convert_pred(self, pred_bbox, org_shape, scores_thresh=0.1):
        result = []
        classes = self.heads['yolo-0'] // 3
        for k in pred_bbox:
            if 'decode' in k:
                result.append(pred_bbox[k].view(-1, classes))
        pred_bbox = torch.cat(result, 0).detach().cpu().numpy()
        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        h, w, _ = org_shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        ration_h = max((h, w)) / self.config.DATASET.INPUT_H
        ration_w = max((h, w)) / self.config.DATASET.INPUT_H
        pred_coor = pred_coor * np.array([[ration_w, ration_h, ration_w, ration_h]])
        pred_coor = pred_coor - np.array([[pad[1][0], pad[0][0], pad[1][0], pad[0][0]]])
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        mask = scores > scores_thresh
        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]
        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        return bboxes


if __name__ == '__main__':
    from config import dark53_yolo,hrnet_yolo
    input = torch.rand([3, 3, 416, 416])
    model=Yolodet(hrnet_yolo)
    out = model(input)
    time.sleep(3)
