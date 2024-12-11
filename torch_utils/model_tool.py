import torch
from torch.utils import data

import numpy as np
import random
import os
from torch.optim import lr_scheduler


class PolyScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

'''以下是设置随机数种子'''
def seed_everything(seed, benchmark=False):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if benchmark:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def get_item(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().item()
    else:
        return data


def to_tensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def separate_bn_param(module: torch.nn.Module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])

    assert len(list(module.parameters())) == len(
        params_decay) + len(params_no_decay)
    return params_no_decay, params_decay


def stat_param_num(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataloader(opt, train_dataset, val_dataset, drop_last=True):
    train_sampler, val_sampler = None, None
    if opt.distributed:
        train_sampler = data.DistributedSampler(train_dataset, seed=opt.seed)
        # only ddp need to div batch to per gpu
        train_batch_sampler = data.BatchSampler(train_sampler, opt.batch_size, drop_last=drop_last)
        train_dataloader = data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                           num_workers=opt.num_workers, pin_memory=opt.pin_memory)

        val_sampler = data.DistributedSampler(val_dataset, seed=opt.seed)
        # val_batch_sampler = data.BatchSampler(val_sampler, opt.batch_size, drop_last=False)
        # val_dataloader = data.DataLoader(val_dataset, batch_sampler=val_batch_sampler,
        #                                  num_workers=opt.num_workers, pin_memory=opt.pin_memory)
        val_dataloader = data.DataLoader(val_dataset, batch_size=opt.batch_size, sampler=val_sampler,
                                         num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    else:
        train_dataloader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=opt.num_workers, pin_memory=opt.pin_memory, drop_last=drop_last)
        val_dataloader = data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                         num_workers=opt.num_workers, pin_memory=opt.pin_memory, drop_last=False)

    return train_dataloader, val_dataloader, train_sampler, val_sampler
