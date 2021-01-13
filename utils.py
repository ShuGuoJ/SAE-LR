import torch
from torch.nn import init
from torch import nn
import os
from scipy.io import loadmat
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, -0.1, 0.1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.uniform_(m.weight, -1, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def loadLabel(path):
    '''
    :param path:
    :return: 训练样本标签， 测试样本标签
    '''
    assert os.path.exists(path), '{},路径不存在'.format(path)
    # keys:{train_gt, test_gt}
    gt = loadmat(path)
    return gt['train_gt'], gt['test_gt']


def train(model, criterion, optimizer, dataLoader, device):
    '''
    :param model: 模型
    :param criterion: 目标函数
    :param optimizer: 优化器
    :param dataLoader: 批数据集
    :return: 已训练的模型，训练损失的均值
    '''
    model.train()
    model.to(device)
    trainLoss = []
    for step, ((spectra, neighbor_region), target) in enumerate(dataLoader):
        spectra, neighbor_region, target = spectra.to(device), neighbor_region.to(device), target.to(device)
        if neighbor_region.ndim == 4: neighbor_region = neighbor_region.view((neighbor_region.shape[0], -1))
        input = torch.cat([spectra, neighbor_region], -1)
        out = model(input)

        loss = criterion(out, target)
        trainLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%5 == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('step:{} loss:{} lr:{}'.format(step, loss.item(), lr))
    return model, float(np.mean(trainLoss))


def test(model, criterion, dataLoader, device):
    model.eval()
    evalLoss, correct = [], 0
    for (spectra, neighbor_region), target in dataLoader:
        spectra, neighbor_region, target = spectra.to(device), neighbor_region.to(device), target.to(device)
        if neighbor_region.ndim == 4: neighbor_region = neighbor_region.view((neighbor_region.shape[0], -1))
        input = torch.cat([spectra, neighbor_region], -1)
        logits = model(input)
        loss = criterion(logits, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(logits, dim=-1)
        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)