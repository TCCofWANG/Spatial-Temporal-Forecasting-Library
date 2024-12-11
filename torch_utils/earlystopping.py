import numpy as np
import torch
import os

# class EarlyStopping:
#     def __init__(self,optimizer,scheduler ,patience=10, verbose=True, delta=0 ):
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#
#
#     def __call__(self, val_loss, model,epoch):
#         score = -val_loss
#         if self.best_score is None:
#             self.best_score = score
#             # self.save_checkpoint(val_loss, model,epoch)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.verbose:
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             # self.save_checkpoint(val_loss, model,epoch)
#             self.counter = 0


class EarlyStopping:
    def __init__(self, path,optimizer,scheduler ,patience=10, verbose=True, delta=0 ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model,epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        save_files = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler.state_dict(),
            'epoch': epoch}
        filepath=os.path.split(self.path)[0] # 将文件夹和文件分开
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        torch.save(save_files, self.path)
        self.val_loss_min = val_loss
