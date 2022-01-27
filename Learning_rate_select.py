# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import random,os
from datetime import datetime
from dataset import CustomDataSet, collate_fn
from model import AttentionDTA
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from tensorboardX import SummaryWriter
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from utils import rmse_f, mse_f, pearson_f, spearman_f, ci_f
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def test_precess(model,pbar):
    loss_f = nn.MSELoss()
    model.eval()
    test_losses = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds,proteins, labels = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            labels = labels.cuda()
            predicts= model.forward(compounds,  proteins)
            loss = loss_f(predicts, labels.view(-1, 1))
            total_preds = torch.cat((total_preds, predicts.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)
            test_losses.append(loss.item())
    Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
    test_loss = np.average(test_losses)
    return Y, P, test_loss, mean_squared_error(Y, P), mean_absolute_error(Y, P), r2_score(Y, P)

def test_model(test_dataset_load,save_path,DATASET,lable = "Train",save = True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(test_dataset_load)),
        total=len(test_dataset_load))
    T, P, loss_test, mse_test, mae_test, r2_test = test_precess(model,test_pbar)
    if save:
        with open(save_path + "/{}_stable_{}_prediction.txt".format(DATASET,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};MSE:{:.5f};MAE:{:.5f};R2:{:.5f}.' \
        .format(lable, loss_test, mse_test, mae_test, r2_test)
    print(results)
    return results,mse_test, mae_test, r2_test


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    """select seed"""
    SEED = 4321
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """Load preprocessed data."""
    DATASET = "KIBA"
    print("Find learning rate on {}".format(DATASET))
    tst_path = './datasets/{}.txt'.format(DATASET)
    with open(tst_path, 'r') as f:
        cpi_list = f.read().strip().split('\n')
    print("load finished")
    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(cpi_list, SEED)
    # random.shuffle(cpi_list)
    Batch_size = 128
    weight_decay = 1e-4
    Learning_rate = 1e-10
    Epoch = 10
    mode = "mutil-head-attention"
    save_path = "./hyperparameter/learning_rate/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + 'The_results.txt'
    dataset = CustomDataSet(dataset)
    dataset_len = len(dataset)
    valid_size = int(0.2 * dataset_len)
    train_size = dataset_len - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dataset_load = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=2,
                                            collate_fn=collate_fn)
    valid_dataset_load = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=False, num_workers=2,
                                            collate_fn=collate_fn)

    """ create model"""
    model = AttentionDTA().cuda()
    """weight initialize"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    LOSS_F = nn.MSELoss()
    optimizer = optim.AdamW(
                [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=Learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10)
    note = "LR"
    writer = SummaryWriter(log_dir=save_path, comment=note)

    """Start training."""
    print('Training...')
    for epoch in range(1, Epoch + 1):
        trian_pbar = tqdm(
            enumerate(
                BackgroundGenerator(train_dataset_load)),
            total=len(train_dataset_load))
        """train"""
        train_losses_in_epoch = []
        model.train()
        for trian_i, train_data in trian_pbar:
            '''data preparation '''
            trian_compounds, trian_proteins, trian_labels = train_data
            trian_compounds = trian_compounds.cuda()
            trian_proteins = trian_proteins.cuda()
            trian_labels = trian_labels.cuda()
            optimizer.zero_grad()
            # 正向传播，反向传播，优化
            predicts = model.forward(trian_compounds, trian_proteins)
            train_loss = LOSS_F(predicts, trian_labels.view(-1, 1))
            train_losses_in_epoch.append(train_loss.item())
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            optimizer.step()

        train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
        current_lr = optimizer.param_groups[0]['lr']
        print("Epoch:{};Loss:{};LR:{}".format(epoch, train_loss_a_epoch, current_lr))
        writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('LR with train_loss_a_epoch', train_loss_a_epoch, current_lr)
        scheduler.step()



