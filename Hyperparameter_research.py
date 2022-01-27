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
import time
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
    print("Hyperparameter research in {}".format(DATASET))
    tst_path = './datasets/{}.txt'.format(DATASET)
    with open(tst_path, 'r') as f:
        cpi_list = f.read().strip().split('\n')
    print("load finished")
    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(cpi_list, SEED)
    weight_decay = 1e-4
    Learning_rate = 5e-5
    Patience = 50
    Epoch = 500
    save_path = "./hyperparameter/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    """seach for head num"""
    best_head_num = 4
    best_result = 100
    for head_num in [12, 10, 8, 6, 4, 2]:
        """Output files."""
        save_path_i = "{}/Head_num/".format(save_path)
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)
        file_results = save_path_i + 'Head_num.txt'
        dataset = CustomDataSet(dataset)
        dataset_len = len(dataset)
        valid_size = int(0.2 * dataset_len)
        test_size = int(0.2 * dataset_len)
        train_size = dataset_len - valid_size - test_size
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size,test_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2,
                                            collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2,
                                            collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2,
                                        collate_fn=collate_fn)
        """ create model"""
        model = AttentionDTA(head_num = head_num).cuda()
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
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=Learning_rate, max_lr=Learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=train_size // 128)
        """Start training."""
        print('Training for head num:{}'.format(model.head_num))
        patience = 0
        best_score = 100
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
                '''前向传播与反向传播'''
                '''梯度置0'''
                optimizer.zero_grad()
                # 正向传播，反向传播，优化
                predicts = model.forward(trian_compounds, trian_proteins)
                train_loss = LOSS_F(predicts, trian_labels.view(-1, 1))
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
            valid_losses_in_epoch = []
            model.eval()
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.cuda()
                    valid_proteins = valid_proteins.cuda()
                    valid_labels = valid_labels.cuda()
                    valid_predictions = model.forward(valid_compounds, valid_proteins)
                    valid_loss = LOSS_F(valid_predictions, valid_labels.view(-1, 1))
                    valid_losses_in_epoch.append(valid_loss.item())
                    total_preds = torch.cat((total_preds, valid_predictions.cpu()), 0)
                    total_labels = torch.cat((total_labels, valid_labels.cpu()), 0)
                Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
            valid_MSE = mean_squared_error(Y, P)
            valid_MAE = mean_absolute_error(Y, P)
            valid_R2 = r2_score(Y, P)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)  # 一次epoch的平均验证loss

            if valid_MSE < best_score:
                best_score = valid_MSE
                patience = 0
                torch.save(model.state_dict(), save_path_i + 'valid_best_checkpoint.pth')
            else:
                patience+=1
            epoch_len = len(str(Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_MSE: {valid_MSE:.5f} ' +
                         f'valid_MAE: {valid_MAE:.5f} ' +
                         f'valid_R2: {valid_R2:.5f} ')
            print(print_msg)

            if patience == Patience:
                break
            torch.save(model.state_dict(), save_path_i + 'stable_checkpoint.pth')
            """load trained model"""
            model.load_state_dict(torch.load(save_path_i + "valid_best_checkpoint.pth"))
        testset_test_results,mse_test, mae_test, r2_test = test_model(test_dataset_load,save_path_i,DATASET,lable="Test")
        with open(file_results, 'a') as f:
            f.write("results on \n".format(head_num))
            f.write(testset_test_results + '\n')
        if mse_test < best_result:
            best_result = mse_test
            best_head_num = head_num

    del model
    torch.cuda.empty_cache()
    time.sleep(5)

    """seach for batch size"""
    best_batch_size = 128
    best_result = 100
    for batch_size in [512,256,128,64,32,16]:
        """Output files."""
        save_path_i = "{}/Batch_size/".format(save_path)
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)
        file_results = save_path_i + 'Batch_size.txt'
        dataset = CustomDataSet(dataset)
        dataset_len = len(dataset)
        valid_size = int(0.2 * dataset_len)
        test_size = int(0.2 * dataset_len)
        train_size = dataset_len - valid_size - test_size
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                   [train_size, valid_size, test_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                        collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2,
                                        collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2,
                                       collate_fn=collate_fn)
        """ create model"""
        model = AttentionDTA(head_num=best_head_num).cuda()
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
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}],
            lr=Learning_rate)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=Learning_rate, max_lr=Learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=train_size // batch_size)
        """Start training."""
        print('Training for batch size:{}'.format(batch_size))
        patience = 0
        best_score = 100
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
                '''前向传播与反向传播'''
                '''梯度置0'''
                optimizer.zero_grad()
                # 正向传播，反向传播，优化
                predicts = model.forward(trian_compounds, trian_proteins)
                train_loss = LOSS_F(predicts, trian_labels.view(-1, 1))
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
            valid_losses_in_epoch = []
            model.eval()
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.cuda()
                    valid_proteins = valid_proteins.cuda()
                    valid_labels = valid_labels.cuda()
                    valid_predictions = model.forward(valid_compounds, valid_proteins)
                    valid_loss = LOSS_F(valid_predictions, valid_labels.view(-1, 1))
                    valid_losses_in_epoch.append(valid_loss.item())
                    total_preds = torch.cat((total_preds, valid_predictions.cpu()), 0)
                    total_labels = torch.cat((total_labels, valid_labels.cpu()), 0)
                Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
            valid_MSE = mean_squared_error(Y, P)
            valid_MAE = mean_absolute_error(Y, P)
            valid_R2 = r2_score(Y, P)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)  # 一次epoch的平均验证loss

            if valid_MSE < best_score:
                best_score = valid_MSE
                patience = 0
                torch.save(model.state_dict(), save_path_i + 'valid_best_checkpoint.pth')
            else:
                patience += 1
            epoch_len = len(str(Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_MSE: {valid_MSE:.5f} ' +
                         f'valid_MAE: {valid_MAE:.5f} ' +
                         f'valid_R2: {valid_R2:.5f} ')
            print(print_msg)

            if patience == Patience:
                break
            torch.save(model.state_dict(), save_path_i + 'stable_checkpoint.pth')
            """load trained model"""
            model.load_state_dict(torch.load(save_path_i + "valid_best_checkpoint.pth"))
        testset_test_results, mse_test, mae_test, r2_test = test_model(test_dataset_load, save_path_i, DATASET,
                                                                       lable="Test")
        with open(file_results, 'a') as f:
            f.write("results on \n".format(batch_size))
            f.write(testset_test_results + '\n')
        if mse_test < best_result:
            best_result = mse_test
            best_batch_size = batch_size

    del model
    torch.cuda.empty_cache()
    time.sleep(5)

    """seach for dropout rate"""
    best_dropout_rate = 0.1
    best_result = 100
    for dropout_rate in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        """Output files."""
        save_path_i = "{}/Dropout_rate/".format(save_path)
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)
        file_results = save_path_i + 'dropout_rate.txt'
        dataset = CustomDataSet(dataset)
        dataset_len = len(dataset)
        valid_size = int(0.2 * dataset_len)
        test_size = int(0.2 * dataset_len)
        train_size = dataset_len - valid_size - test_size
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                   [train_size, valid_size, test_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True, num_workers=2,
                                        collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2,
                                        collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2,
                                       collate_fn=collate_fn)
        """ create model"""
        model = AttentionDTA(head_num=best_head_num,dropout_rate = dropout_rate).cuda()
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
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}],
            lr=Learning_rate)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=Learning_rate, max_lr=Learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=train_size // best_batch_size)
        """Start training."""
        print('Training for dropout rate:{}'.format(model.dropout_rate))
        patience = 0
        best_score = 100
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
                '''前向传播与反向传播'''
                '''梯度置0'''
                optimizer.zero_grad()
                # 正向传播，反向传播，优化
                predicts = model.forward(trian_compounds, trian_proteins)
                train_loss = LOSS_F(predicts, trian_labels.view(-1, 1))
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss
            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
            valid_losses_in_epoch = []
            model.eval()
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.cuda()
                    valid_proteins = valid_proteins.cuda()
                    valid_labels = valid_labels.cuda()
                    valid_predictions = model.forward(valid_compounds, valid_proteins)
                    valid_loss = LOSS_F(valid_predictions, valid_labels.view(-1, 1))
                    valid_losses_in_epoch.append(valid_loss.item())
                    total_preds = torch.cat((total_preds, valid_predictions.cpu()), 0)
                    total_labels = torch.cat((total_labels, valid_labels.cpu()), 0)
                Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
            valid_MSE = mean_squared_error(Y, P)
            valid_MAE = mean_absolute_error(Y, P)
            valid_R2 = r2_score(Y, P)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)  # 一次epoch的平均验证loss

            if valid_MSE < best_score:
                best_score = valid_MSE
                patience = 0
                torch.save(model.state_dict(), save_path_i + 'valid_best_checkpoint.pth')
            else:
                patience += 1
            epoch_len = len(str(Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_MSE: {valid_MSE:.5f} ' +
                         f'valid_MAE: {valid_MAE:.5f} ' +
                         f'valid_R2: {valid_R2:.5f} ')
            print(print_msg)

            if patience == Patience:
                break
            torch.save(model.state_dict(), save_path_i + 'stable_checkpoint.pth')
            """load trained model"""
            model.load_state_dict(torch.load(save_path_i + "valid_best_checkpoint.pth"))
        testset_test_results, mse_test, mae_test, r2_test = test_model(test_dataset_load, save_path_i, DATASET,
                                                                       lable="Test")
        with open(file_results, 'a') as f:
            f.write("results on \n".format(dropout_rate))
            f.write(testset_test_results + '\n')
        if mse_test < best_result:
            best_dropout_rate = dropout_rate
    print("best head num is {}; best batch size is {}; best dropout rate is {};".format(best_head_num,best_batch_size,best_dropout_rate))





