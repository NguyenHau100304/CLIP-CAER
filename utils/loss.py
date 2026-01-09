# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal

class LSR2(nn.Module):
    def __init__(self,e,label_mode):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.label_mode = label_mode

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        mask = (one_hot==0)
        balance_weight = torch.tensor([0.065267810,0.817977729,1.035884371,0.388144355,0.19551041668]).to(one_hot.device)
        ex_weight = balance_weight.expand(one_hot.size(0),-1)
        resize_weight = ex_weight[mask].view(one_hot.size(0),-1)
        resize_weight /= resize_weight.sum(dim=1, keepdim=True)
        one_hot[mask] += (resize_weight*smooth_factor).view(-1)
        return one_hot.to(target.device)
    
    def forward(self, x, target):
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)

class BlvLoss(nn.Module):
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_list)) - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name

    def forward(self, pred, target):
        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)
        pred = pred + (viariation.abs() / self.frequency_list.max() * self.frequency_list)
        loss = F.cross_entropy(pred, target, reduction='none')

        return loss.mean()
    

import math
import numpy as np

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        LDAM Loss: Label-Distribution-Aware Margin Loss
        Args:
            cls_num_list (list): Danh sách số lượng mẫu của mỗi lớp [1383, 129, 30, 157, 422]
            max_m (float): Margin lớn nhất (thường là 0.5)
            weight (tensor): Trọng số cho loss (nếu muốn kết hợp re-weighting)
            s (float): Scaling factor (thường là 30)
        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda()
        self.m_list = m_list
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

def get_cb_weights(cls_num_list, beta=0.9999):
    """
    Tính toán trọng số Class-Balanced (CB) thay vì Inverse Frequency đơn thuần.
    Dùng cho tham số 'weight' trong LDAMLoss hoặc CrossEntropy.
    """
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(cls_num_list)
    return torch.FloatTensor(weights).cuda()