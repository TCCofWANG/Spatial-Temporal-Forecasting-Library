import torch
import torch.distributed as dist

from collections import defaultdict # 内置字典，提供默认值
from .dist import is_dist_avail_and_initialized, get_world_size, is_master_process
from .model_tool import get_item
import time
import datetime
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeterLogger(object):
    def __init__(self, names=None):
        self.meters = defaultdict(AverageMeter)
        self.world_size = get_world_size()
        self.is_master = is_master_process()

        if names is not None:
            for name in names:
                self.meters[name] = AverageMeter()

        self.log_stats = {}

    def add_meter(self, name):
        self.meters[name] = AverageMeter()

    def reset(self):
        for meter in self.meters.values():
            meter.reset()
        self.log_stats = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            v = get_item(v)
            self.meters[k].update(v, n=1)

    def values(self):
        value_dict = {k: v.val for k, v in self.meters.items()}
        return value_dict

    def averages(self):
        avg_dict = {k: v.avg for k, v in self.meters.items()}
        return avg_dict

    def get_finish_epoch_logs(self):
        if not self.log_stats:
            raise ValueError('A epoch is not finished.')
        return self.log_stats

    def log_every(self, data_loader, header='', desc=''):

        MB = 1024.0 * 1024.0

        _iter = tqdm(iter(data_loader), desc=desc) if self.is_master else data_loader
        start = time.time()
        for obj in _iter:
            yield obj

            value_dict = self.values()
            # self.log_stats = value_dict
            if torch.cuda.is_available():
                value_dict['memory'] = round(torch.cuda.max_memory_allocated() / MB, 1)
            if self.is_master:
                _iter.set_postfix(**value_dict) # 设置tqdm的显示

        # 各进程同步
        sync_value_dict = self.synchronize_between_processes()

        for name, value in sync_value_dict.items():
            self.log_stats[name] = value

        total_time = time.time() - start
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log_stats['total_time'] = total_time_str
        self.log_stats['iter_cnt'] = len(data_loader)

        print('{} | total time: {} ({} s / it)'.format(header, total_time_str, round(total_time / len(data_loader), 4)))
        print('{} | {}'.format(header, str(self)))

    def synchronize_between_processes(self, ops='avg'):
        meter_values = self.averages()
        if not is_dist_avail_and_initialized():
            return meter_values

        for name, value in meter_values.items():
            value = torch.tensor(value).cuda()
            dist.barrier()
            dist.all_reduce(value)

            meter_values[name] = get_item(value) / self.world_size

        return meter_values

    def __str__(self):
        _str = []
        keys = sorted(self.log_stats)
        for k in keys:
            _str.append('{} : {}'.format(k, self.log_stats[k]))
        return ' | '.join(_str)
