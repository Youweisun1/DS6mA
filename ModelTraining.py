
import sys
import random
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
sys.setrecursionlimit(15000)#15000
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, accuracy_score
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
import os
import matplotlib
from sklearn import metrics
matplotlib.use('Agg')
from data_processing import load_data
from MyModel import DS6mA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
model = DS6mA()
np.random.seed(1377)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 60
Margin = 2
SAVE_DIR = 'DS6mA'
if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

filename = 'data/A.thaliana.xlsx'    # change directory here to apply different datasets.

[X_train, y_train, X_valid, y_valid, X_test, y_test] = load_data(filename)

X_train = np.concatenate((X_train, X_valid), axis=0)
y_train = torch.cat((y_train, y_valid), axis=0)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)
batch_size = 64

train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=Margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(self.margin - euclidean_distance, 2))

        return loss_contrastive


def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)
    indices = np.random.permutation(batch_size).tolist()
    shuffled_batch = [batch[i] for i in indices]  

    for i in range(int(batch_size)-1):
        seq1, label1 = shuffled_batch[i][0], shuffled_batch[i][1]
        for j in range(i+1, int(batch_size)):
            seq2, label2 = shuffled_batch[j][0], shuffled_batch[j][1]
            label1_ls.append(label1.unsqueeze(0))
            label2_ls.append(label2.unsqueeze(0))
            label = (label1 ^ label2)  # 异或, 相同为 0 ,相异为 1
            seq1_ls.append(seq1.unsqueeze(0))
            seq2_ls.append(seq2.unsqueeze(0))
            label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)

    return seq1, seq2, label, label1, label2
train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
net = DS6mA().to(device)
lr = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
criterion = ContrastiveLoss()
criterion_model = nn.CrossEntropyLoss(reduction='mean')
best_auc = 0

all_epoch_loss = []
for epoch in range(EPOCH):
    loss_ls = []
    loss1_ls = []
    loss2_3_ls = []
    t0 = time.time()
    net.train()
    for seq1, seq2, label, label1, label2 in train_iter_cont:
        output1 = net(seq1)
        output2 = net(seq2)
        output3 = net.trainModel(seq1)
        output4 = net.trainModel(seq2)
        loss1 = criterion(output1, output2, label)
        loss2 = criterion_model(output3, label1)
        loss3 = criterion_model(output4, label2)
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ls.append(loss.item())
        loss1_ls.append(loss1.item())
        loss2_3_ls.append((loss2 + loss3).item())
    net.eval()
    outputs = net.trainModel(X_test.to(device))
    outputs = outputs.cpu().detach().numpy()
    predict = np.vstack(outputs[:, 1])
    auc = metrics.roc_auc_score(y_test, predict)
    all_epoch_loss.append(np.mean(loss_ls))
