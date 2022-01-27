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

def get_kfold_data(i, datasets, k=5):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(datasets) // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] # 若不能整除，将多的case放在最后一折里
        trainset = datasets[0:val_start]

    return trainset, validset

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
    DATASET = "Davis"
    # DATASET = "Metz"
    # DATASET = "KIBA"
    print("Train in {}".format(DATASET))
    tst_path = './datasets/{}.txt'.format(DATASET)
    with open(tst_path, 'r') as f:
        cpi_list = f.read().strip().split('\n')
    print("load finished")
    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(cpi_list, SEED)
    # random.shuffle(cpi_list)
    K_Fold = 5
    Batch_size = 128
    weight_decay = 1e-4
    Learning_rate = 5e-5
    Patience = 50
    Epoch = 500
    """Output files."""
    save_path = "./Results/{}/".format(DATASET)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + 'The_results.txt'

    MSE_List, MAE_List, R2_List = [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, '第', i_fold + 1, '折', '*' * 25)
        trainset, testset = get_kfold_data(i_fold, dataset, k=K_Fold)
        TVdataset = CustomDataSet(trainset)
        test_dataset = CustomDataSet(testset)
        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=2,
                                        collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=False, num_workers=2,
                                        collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=2,
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
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=Learning_rate, max_lr=Learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=train_size // Batch_size)
        save_path_i = "{}/{}_Fold/".format(save_path, i_fold + 1)
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)
        note = ""
        writer = SummaryWriter(log_dir=save_path_i, comment=note)

        """Start training."""
        print('Training...')
        start = timeit.default_timer()
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

                optimizer.zero_grad()

                predicts = model.forward(trian_compounds, trian_proteins)
                train_loss = LOSS_F(predicts, trian_labels.view(-1, 1))
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()

                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)
            writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)

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
            writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
            writer.add_scalar('Valid MSE', valid_MSE, epoch)
            writer.add_scalar('Valid MAE', valid_MAE, epoch)
            writer.add_scalar('Valid R2', valid_R2, epoch)

            if patience == Patience:
                break
        torch.save(model.state_dict(), save_path_i + 'stable_checkpoint.pth')
        """load trained model"""
        model.load_state_dict(torch.load(save_path_i + "valid_best_checkpoint.pth"))
        trainset_test_results,_,_,_ = test_model(train_dataset_load, save_path_i, DATASET, lable="Train")
        validset_test_results,_,_,_ = test_model(valid_dataset_load, save_path_i, DATASET, lable="Valid")
        testset_test_results,mse_test, mae_test, r2_test = test_model(test_dataset_load,save_path_i,DATASET,lable="Test")
        with open(save_path + "The_results.txt", 'a') as f:
            f.write("results on {}th fold\n".format(i_fold+1))
            f.write(trainset_test_results + '\n')
            f.write(validset_test_results + '\n')
            f.write(testset_test_results + '\n')
        writer.close()
        MSE_List.append(mse_test)
        MAE_List.append(mae_test)
        R2_List.append(r2_test)
    MSE_mean, MSE_var = np.mean(MSE_List), np.sqrt(np.var(MSE_List))
    MAE_mean, MAE_var = np.mean(MAE_List), np.sqrt(np.var(MAE_List))
    R2_mean, R2_var = np.mean(R2_List), np.sqrt(np.var(R2_List))
    with open(save_path + 'The_results.txt', 'a') as f:
        f.write('The results on {}:'.format(DATASET) + '\n')
        f.write('MSE(std):{:.4f}({:.4f})'.format(MSE_mean, MSE_var) + '\n')
        f.write('MAE(std):{:.4f}({:.4f})'.format(MAE_mean, MAE_var) + '\n')
        f.write('R2(std):{:.4f}({:.4f})'.format(R2_mean, R2_var) + '\n')
    print('MSE(std):{:.4f}({:.4f})'.format(MSE_mean, MSE_var))
    print('MAE(std):{:.4f}({:.4f})'.format(MAE_mean, MAE_var))
    print('R2(std):{:.4f}({:.4f})'.format(R2_mean, R2_var))




