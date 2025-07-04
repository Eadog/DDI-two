import random
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from model1 import D1Model1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from dataset import *
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from time import strftime
from utils import *
import datetime
import logging
import os

# def seed_everything(seed=927):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    pred, label, prob = [], [], []
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        y = data[0].y.view(-1, 1).float().to(device)
        #y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1,data2)
        #output,hidden_states = model(data1,data2)
        pred.append(output.detach().sigmoid().cpu())
        label.extend(data1.y.detach().cpu().numpy())
        #breakpoint()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx,
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                          loss.item()))
   
    scheduler.step()
def test(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    pred, label, prob = [], [], []
    
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0].to(device)
            data2 = data[1].to(device)
            outputs = model(data1,data2)
            prob.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            binary_pred = (torch.sigmoid(outputs) > 0.5).int()
            pred.extend(binary_pred.detach().cpu().numpy())
            label.extend(data1.y.detach().cpu().numpy())
   
    return np.array(label), np.array(pred), np.array(prob)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeePGG")
    parser.add_argument('--saved_root', default='./trained_record/xiaorongZhang/xulie/', type=str, required=False,
                        help='..')
    parser.add_argument('--batch_size', default=512, type=int, required=False,
                        help='..')
    parser.add_argument('--epochs', default=200, type=int, required=False,
                        help='..')
    parser.add_argument('--lr', default=5e-4, type=float, required=False,
                        help='..')
    parser.add_argument('--weight_decay', default=1e-2, type=float, required=False,
                        help='..')
    parser.add_argument('--gamma', default=0.8, type=float, required=False,
                        help='..')
    parser.add_argument('--dropout', default=0.3, type=float, required=False,
                        help='..')
    parser.add_argument('--step_size', default=10, type=int, required=False,
                        help='..')
    parser.add_argument('--num_class', default=2, type=int, required=False,
                        help='..')
    parser.add_argument('--mode', default='train', type=str, required=False,
                        help='train or test')
    parser.add_argument('--model_name', default='drugbank', type=str, required=False,
                        help='..')
    parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), type=torch.tensor, required=False,
                        help='..')
    args = parser.parse_args()

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
    print('Learning rate: ', args.lr)
    print('Epochs: ',args.epochs)
    print('batch_size: ',args.batch_size)
    datafile = 'Zhang'
    # CPU or GPU
    drug1_data = DDIDataset(root='data/', dataset=datafile + '_drug1')
    drug2_data = DDIDataset(root='data/', dataset=datafile + '_drug2')

    lenth = len(drug1_data)
    pot = int(lenth/5)
    print('lenth', lenth)
    print('pot', pot)

    nowtime = strftime('%Y-%m-%d-%H:%M:%S')
    saved_path = args.saved_root + nowtime + '/'
    log_filename = '{0}{1}五折1.log'.format(saved_path, nowtime)
    if not os.path.exists(args.saved_root):
        os.makedirs(args.saved_root)
        print("Directory ", args.saved_root, " Created ")
    else:
        print("Directory ", args.saved_root, " already exists")
    # in saved path directory, make a nowtime directory
    if not os.path.exists(args.saved_root + nowtime):
        os.makedirs(args.saved_root + nowtime)
        print("Directory ", args.saved_root + nowtime, " Created ")
    else:
        print("Directory ", args.saved_root + nowtime, " already exists")
    #seed_everything()
    #5:1划分数据集(训练集:测试集)
    random.seed(42)
    random_num = random.sample(range(0, lenth), lenth)
    for i in range(5):
        test_num = random_num[pot*i:pot*(i+1)]
        train_num = random_num[:pot*i] + random_num[pot*(i+1):]

        drug1_data_train = [data for data in drug1_data[train_num]]
        drug1_data_test = [data for data in drug1_data[test_num]]
        drug1_loader_train = DataLoader(drug1_data_train, batch_size=args.batch_size)
        drug1_loader_test = DataLoader(drug1_data_test, batch_size=args.batch_size)

        drug2_data_test = [data for data in drug2_data[test_num]]
        drug2_data_train = [data for data in drug2_data[train_num]]
        drug2_loader_train = DataLoader(drug2_data_train, batch_size=args.batch_size)
        drug2_loader_test = DataLoader(drug2_data_test, batch_size=args.batch_size)

        model = D1Model1(args)
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        starttime = datetime.datetime.now()
        best_auc = 0
        last_epoch_time = starttime
        best_epoch = 0
        model.best_metric = -1.0
        model.best_ma_f1 = -1.0
        model.best_roc_auc = -1.0
        model.best_pr_auc = -1.0

        log = f"第{i}折" + "\n"
        get_logging(log, log_filename, eval='test')
        for epoch in range(args.epochs):
            train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
            #breakpoint()
            label, pred, prob = test(model, device, drug1_loader_test, drug2_loader_test)
            # compute preformence

            acc = metrics.accuracy_score(label, pred)
            auroc = metrics.roc_auc_score(label, prob)
            f1 = metrics.f1_score(label, pred, zero_division=0)
            ap = metrics.average_precision_score(label, prob)
            p = metrics.precision_score(label, pred, zero_division=0)
            r = metrics.recall_score(label, pred, zero_division=0)


            best_epoch = 0
            path = f"{saved_path}{i}_DDinter2.pt"
            if acc > model.best_acc:
                model.best_acc = acc
                model.best_epoch = epoch  # 记录最佳指标对应的 epoch
                model.best_roc_auc = auroc
                model.best_f1 = f1
                model.best_ap = ap
                #print("acc值：", model.best_balanced_acc)
                torch.save(model.state_dict(), path)

            #breakpoint()
            print(f"test dataset acc: {acc}")
            print(f"test dataset roc_auc: {auroc}")
            print(f"test dataset f1: {f1}")
            print(f"test dataset ap: {ap}")

            log = f"epoch: {epoch}" + "\n" + \
                f"test dataset acc: {acc}" + "\n" + \
                f"test dataset roc_auc: {auroc}" + "\n" + \
                f"test dataset macro_f1: {f1}" + "\n" + \
                f"test dataset ap: {ap}" + "\n" 
            get_logging(log, log_filename, eval='test')
        
        print(f"best_acc: {model.best_acc}")
        print(f"best_roc_auc: {model.best_roc_auc}")
        print(f"best_f1: {model.best_f1}")
        print(f"best_ap: {model.best_ap}")

        log = f"best_acc: {model.best_acc}" + "\n" + \
            f"best_roc_auc: {model.best_roc_auc}" + "\n" + \
            f"best_f1: {model.best_f1}" + "\n" + \
            f"best_ap: {model.best_ap}" + "\n"
        get_logging(log, log_filename, eval='test')
        torch.cuda.empty_cache()
    # get_logging(log, log_filename, eval='test')



