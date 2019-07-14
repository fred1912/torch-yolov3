import torch
import torch.nn as nn
import numpy as np
def _sigmoid(x):
  y = torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)
  return y

class yoloDecode(nn.Module):

    def __init__(self,id,config):

        super(yoloDecode,self).__init__()
        self.scale = config.DATASET.DOWN_RATION[id]
        self.H=config.DATASET.input_h //self.scale
        self.W=config.DATASET.input_w // self.scale
        self.anchors=torch.from_numpy(np.array(config.DATASET.ANCHOR[id])).float()
        y = torch.arange(self.H)[np.newaxis, :].expand([self.W, self.H])
        x = torch.arange(self.W)[:, np.newaxis].expand([self.W, self.H])
        xy_grid = torch.cat([x[:, :, np.newaxis], y[:, :, np.newaxis]], dim=-1)[np.newaxis, :, :, np.newaxis, :]
        self.xy_grid=xy_grid.float()

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
        pred_xy = (_sigmoid(pred_conv_dxdy) + self.xy_grid) * self.scale
        pred_wh = (torch.exp(pred_conv_dwdh) * self.anchors) * self.scale
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = _sigmoid(pred_conv_conf)
        pred_prob = _sigmoid(pred_conv_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)
        return pred_bbox

def convert_pred(pred_bbox, org_img_shape, valid_scale = (0, np.inf)):
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    org_h, org_w = org_img_shape

    # (3)将预测的bbox中超出原图的部分裁掉
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    # (4)将无效bbox的coor置为0
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4)去掉不在有效范围内的bbox
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (4)将score低于score_threshold的bbox去掉
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > 0.3
    mask = np.logical_and(scale_mask, score_mask)

    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]
    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    return bboxes
