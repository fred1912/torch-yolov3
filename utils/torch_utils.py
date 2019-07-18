import torch
def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-7, max=1-1e-7)
    #y = x.sigmoid()
    return y

def cal_IOU(boxes1, boxes2):
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                      boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                      boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    inter_max_xy = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_min_xy = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    union = area1 + area2 - inter_area
    ious = inter_area / union
    ious = torch.clamp(ious, min=0., max=1.0)
    return ious

def cal_GIOU(boxes1, boxes2):
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    inter_max_xy = torch.min(boxes1[..., 2:],boxes2[..., 2:])
    inter_min_xy = torch.max(boxes1[..., :2],boxes2[..., :2])
    out_max_xy = torch.max(boxes1[..., 2:],boxes2[..., 2:])
    out_min_xy = torch.min(boxes1[..., :2],boxes2[..., :2])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[..., 0] * outer[..., 1]
    union = area1+area2-inter_area
    closure = outer_area
    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    return ious
