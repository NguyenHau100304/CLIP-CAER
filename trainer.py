# trainer.py
import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.utils import AverageMeter, ProgressMeter

class Trainer:
    """A class that encapsulates the training and validation logic."""
    def __init__(self, model, criterion, optimizer, scheduler, device,log_txt_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.print_freq = 10
        self.log_txt_path = log_txt_path

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        """Runs one epoch of training or validation."""
        if is_train:
            self.model.train()
            prefix = f"Train Epoch: [{epoch_str}]"
        else:
            self.model.eval()
            prefix = f"Valid Epoch: [{epoch_str}]"

        losses = AverageMeter('Loss', ':.4e')
        war_meter = AverageMeter('WAR', ':6.2f')
        progress = ProgressMeter(
            len(loader), 
            [losses, war_meter], 
            prefix=prefix, 
            log_txt_path=self.log_txt_path  
        )

        all_preds = []
        all_targets = []

        context = torch.enable_grad() if is_train else torch.no_grad()
        
        with context:
            for i, (images_face, images_body, target) in enumerate(loader):
                images_face = images_face.to(self.device)
                images_body = images_body.to(self.device)
                target = target.to(self.device)

                # Forward pass
                output = self.model(images_face, images_body)
                loss = self.criterion(output, target)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Record metrics
                preds = output.argmax(dim=1)
                correct_preds = preds.eq(target).sum().item()
                acc = (correct_preds / target.size(0)) * 100.0

                losses.update(loss.item(), target.size(0))
                war_meter.update(acc, target.size(0))

                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

                if i % self.print_freq == 0:
                    progress.display(i)
        
        # Calculate epoch-level metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
        war = war_meter.avg # Weighted Average Recall (WAR) is just the overall accuracy
        
        # Unweighted Average Recall (UAR)
        class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6) # Add epsilon to avoid division by zero
        uar = np.nanmean(class_acc) * 100

        logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
        with open(self.log_txt_path, 'a') as f:
            f.write('Current UAR: {war:.3f}'.format(war=war) + '\n')
            f.write('Current UAR: {uar:.3f}'.format(uar=uar) + '\n')
        return war, uar, losses.avg, cm
        
    def train_epoch(self, train_loader, epoch_num):
        """Executes one full training epoch."""
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)
    
    def validate(self, val_loader, epoch_num_str="Final"):
        """Executes one full validation run."""
        return self._run_one_epoch(val_loader, epoch_num_str, is_train=False)