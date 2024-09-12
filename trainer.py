import os
import math
import datetime
import numpy as np
from tqdm import tqdm
import os.path as osp
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,recall_score,accuracy_score, precision_score
import torch.optim.lr_scheduler as lr_scheduler

import torch


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, start_epoch, max_epoch, early_stopping_monitor, val_loader,
                 train_loader, test_loader, lr_policy):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.checkpoint = checkpoint
        if not osp.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        self.LR_policy = lr_policy
        self.epoch = 0
        self.r = 0.5
        self.f1 = 0
        self.accuracy = 0
        self.precision = 0
        self.state_best = None
        self.early_stopping_monitor = early_stopping_monitor
        self.val_loss_min = np.Inf
        self.patience_counter = 0
        self.scheduler = None
        self.initialize_scheduler()

    def initialize_scheduler(self):
        """Initializes the scheduler based on the learning rate policy."""
        if self.LR_policy == 'CosineAnnealingWarmRestarts':
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=2, eta_min=0.00001)
        # elif self.LR_policy not in ['poly', 'step', 'warm-up-epoch', 'warm-up-step']:
        #     raise ValueError(f'Unknown lr policy {self.LR_policy}')

    def get_triangle_lr(self, base_lr, max_lr, total_steps, cur, ratio=0.5, annealing_decay=1e-2, momentums=[0.95, 0.85]):
        first = int(total_steps * ratio)
        last = total_steps - first
        min_lr = base_lr * annealing_decay

        cycle = np.floor(1 + cur / total_steps)
        x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)
        if cur < first:
            lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
        else:
            lr = ((base_lr - min_lr) * cur + min_lr * first - base_lr * total_steps) / (first - total_steps)
        if isinstance(momentums, int):
            momentum = momentums
        else:
            if cur < first:
                momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1. - x)
            else:
                momentum = momentums[0]

        return lr, momentum

    def adjust_learning_rate(self, i_batch=None):
        """Adjusts the learning rate according to the chosen policy."""
        db_size = len(self.train_loader)
        total_steps = self.max_epoch * db_size

        if self.scheduler:
            self.scheduler.step()
        else:
            epoch_position = self.epoch + (i_batch / db_size if i_batch is not None else 0)

            if self.LR_policy == 'poly' or self.LR_policy == 'step' or self.LR_policy == 'warm-up-epoch':
                if i_batch == 0:

                    if self.LR_policy == 'poly':
                        lr = self.optimizer.param_groups[0]['lr']
                        max_lr = lr
                        lr = max_lr * (1 - self.epoch / self.max_epoch) ** 0.9
                    elif self.LR_policy == 'step':
                        step_size = 10  # Can be adjusted or passed as an argument.
                        gamma = 0.1  # Can be adjusted or passed as an argument.
                        lr = self.optimizer.param_groups[0]['lr'] * (gamma ** (self.epoch // step_size))
                    elif self.LR_policy == 'warm-up-epoch':
                        max_lr = 0.03  # This value can be adjusted or passed as an argument.
                        # Apply warm-up-epoch policy
                        lr = (1 - abs((self.epoch + 1) / (self.max_epoch + 1) * 2 - 1)) * max_lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
            elif self.LR_policy == 'warm-up-step':
                max_lr = 0.008  # This value can be adjusted or passed as an argument.
                niter = int(epoch_position * db_size)
                lr, _ = self.get_triangle_lr(self.optimizer.param_groups[0]['lr'], max_lr, total_steps, niter)
            # else:
            #     raise ValueError(f'Unknown lr policy {self.LR_policy}')

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr


    def train(self):
        """training the model"""
        self.model.to(self.device)
        self.criterion.to(self.device)
        for epoch in range(self.start_epoch, self.max_epoch):
            self.model.train()
            self.epoch = epoch
            total_loss = 0
            total_correct = 0
            total_samples = 0
            total_true_positives = 0
            total_false_positives = 0

            for i_batch, sample_batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}"):
                X_data = sample_batch["data"].float().to(self.device)
                label = sample_batch["label"].float().to(self.device)
                self.optimizer.zero_grad()
                label_p = self.model(X_data)
                loss = self.criterion(label_p, label)

                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')

                predicted = (label_p > 0.5).float()
                correct = predicted.eq(label.view_as(predicted)).sum().item()
                total_correct += correct
                total_samples += label.size(0)

                true_positives = (predicted * label).sum()
                false_positives = (predicted * (1 - label)).sum()
                total_true_positives += true_positives.item()
                total_false_positives += false_positives.item()

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if self.LR_policy is not None:
                    self.adjust_learning_rate(i_batch=i_batch)
                # print("epoch/i_batch: {}/{}---loss: {:.4f}---lr: {:.5f}".format(
                #     self.epoch, i_batch, loss.item(), self.optimizer.param_groups[0]['lr']))

            average_loss = total_loss / len(self.train_loader)
            if total_samples > 0:
                accuracy = total_correct / total_samples
            else:
                accuracy = 0
            if total_true_positives + total_false_positives > 0:
                precision = total_true_positives / (total_true_positives + total_false_positives)
            else:
                precision = 0.0
            print("Epoch: {}---lr: {:.5f}---Average Loss: {:.4f}---Accuracy: {:.4f}---Precision: {:.4f}".format(
                self.epoch, self.optimizer.param_groups[0]['lr'], average_loss, accuracy, precision))

            # 验证
            val_loss = self.validate()
            if val_loss < self.val_loss_min:
                self.val_loss_min = val_loss
                self.early_stopping_monitor_counter = 0
                self.state_best = self.model.state_dict()
            else:
                self.early_stopping_monitor_counter += 1
            if self.early_stopping_monitor_counter >= self.early_stopping_monitor:
                print("Early stopping triggered.")
                break

        return self.r, self.f1, self.state_best, self.accuracy, self.precision

    def validate(self):
        """validate the performance of the trained model."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        label_p_all = []
        label_t_all = []
        for i_batch, sample_batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Validating"):
            X_data = sample_batch["data"].float().to(self.device)
            label = sample_batch["label"].float().to(self.device)
            with torch.no_grad():
                label_p = self.model(X_data)
                loss = self.criterion(label_p, label)
                total_loss += loss.item()
                total_samples += label.size(0)

                label_p_all.append(label_p.view(-1))
                label_t_all.append(label.view(-1))
                # label_p_all.append(label_p.view(-1).data.cpu().numpy()[0])
                # label_t_all.append(label.view(-1).data.cpu().numpy()[0])

        label_p_all = torch.cat(label_p_all).cpu().numpy()
        label_t_all = torch.cat(label_t_all).cpu().numpy()
        # 计算性能指标
        label_p_binary = (label_p_all > 0.5).astype(int)
        # label_p_binary = [int(x > 0.5) for x in label_p_all]
        r = recall_score(label_t_all, label_p_binary)
        f1 = f1_score(label_t_all, label_p_binary)
        accuracy = accuracy_score(label_t_all, label_p_binary)
        precision = precision_score(label_t_all, label_p_binary)
        if self.f1 < f1 or (self.f1 == f1 and (self.accuracy < accuracy or self.precision < precision)):
            self.f1 = f1
            self.r = r
            self.accuracy = accuracy
            self.precision = precision
            self.state_best = self.model.state_dict()
        average_val_loss = total_loss / total_samples if total_samples > 0 else 0
        print("Validation Loss: {:.6f}\t\tval_r: {:.3f}\tval_f1: {:.3f}\tval_accuracy: {:.3f}\tval_precision: {:.3f}\n".format(average_val_loss, r, f1, accuracy, precision))

        return average_val_loss

    def test(self):
        """validate the performance of the trained model."""
        self.model.eval()
        label_p_all = []
        label_t_all = []
        for i_batch, sample_batch in enumerate(self.test_loader):
            X_data = sample_batch["data"].float().to(self.device)
            label = sample_batch["label"].float().to(self.device)
            with torch.no_grad():
                label_p = self.model(X_data)
            label_p_all.append(label_p.view(-1).data.cpu().numpy()[
                                   0])
            label_t_all.append(label.view(-1).data.cpu().numpy()[0])

        r = recall_score(label_t_all, [int(x > 0.5) for x in
                                       label_p_all])
        f1 = f1_score(label_t_all, [int(x > 0.5) for x in label_p_all])
        if (self.r + self.f1) < (r + f1):
            self.f1 = f1
            self.r = r
            self.state_best = self.model.state_dict()
        print("r: {:.3f}\tf1: {:.3f}\n".format(r, f1))

    def predict(self, loader):
        self.model.to(self.device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for sample_batch in loader:
                X_data = sample_batch["data"].float().to(self.device)
                output = self.model(X_data)
                predictions.append(output.data.cpu().numpy())
        return np.concatenate(predictions, axis=0)

