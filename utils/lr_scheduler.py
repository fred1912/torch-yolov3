from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch.optim.optimizer import Optimizer
from easydict import EasyDict

class CosineLr():

    def __init__(self,optimizer:Optimizer,config:EasyDict,step_per_epoch:int=0):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        if not isinstance(config,EasyDict):
            raise TypeError('{} is not an config'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self.warmup_steps=config['WARMUP_EPOCH']*step_per_epoch
        self.train_steps=config['EPOCH']*step_per_epoch
        self.lr_init=config['INIT_LR']
        self.lr_end=config['END_LR']
        self.global_step=0

    def step(self,lr_set=None):

        self.global_step += 1
        if not lr_set:
            lr=self.__get_lr()
        else:
            lr=lr_set
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def __get_lr(self):

        if self.global_step < self.warmup_steps :
            lr=self.global_step / self.warmup_steps * self.lr_init
        else:
            lr= self.lr_end + 0.5 *(self.lr_init - self.lr_end) * \
                ( 1 + np.cos(np.pi*(self.global_step - self.warmup_steps)/(self.train_steps-self.warmup_steps)))

        return lr

if __name__ == '__main__':

    from config import OPTIM,MODEL
    from torch.optim import SGD
    from tensorboardX import SummaryWriter
    from models.hrnet import get_Hrnet

    base=get_Hrnet(MODEL)
    opti=SGD(base.parameters(),lr=OPTIM['INIT_LR'])
    lr_jist=CosineLr(opti,OPTIM,5000)
    writer=SummaryWriter(comment='lr_just',logdir='../logs')

    for i in range(OPTIM['EPOCH']):
        for j in range(5000):
            lr=lr_jist.step()
            writer.add_scalar('lr',lr,lr_jist.global_step)
    writer.close()

