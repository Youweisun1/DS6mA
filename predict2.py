from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, accuracy_score
import torch.nn.utils.rnn as rnn_utils
import time
import torch
from termcolor import colored
import os
import matplotlib
import numpy as np
from sklearn import metrics
matplotlib.use('Agg')
from data_processing import load_data
from MyModel import mycnn
import csv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def zhibiao(predict_proba, predict_class, Y_test_array):

    predict_proba = predict_proba
    Y_test_array = Y_test_array.reshape([len(Y_test_array),-1]).tolist()
    predict_class = predict_class.reshape([len(predict_class),-1]).tolist()


    # binary evaluate
    acc = accuracy_score(Y_test_array, predict_class)
    binary_acc = metrics.accuracy_score(Y_test_array, predict_class)
    precision = metrics.precision_score(Y_test_array, predict_class)
    recall = metrics.recall_score(Y_test_array, predict_class)
    f1 = metrics.f1_score(Y_test_array, predict_class)
    auc = metrics.roc_auc_score(Y_test_array, predict_proba)
    mcc = metrics.matthews_corrcoef(Y_test_array, predict_class)
    TN, FP, FN, TP = metrics.confusion_matrix(Y_test_array, predict_class).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)
    return acc, auc,mcc,sensitivity,specificity
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
MODEL_SAVE_PATH = 'Mymodel/A.thaliana.pl'
model = mycnn().to(device)


checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cuda:0')
model.load_state_dict(checkpoint['model'])


filename = 'data/A.thaliana.xlsx'    

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
model.eval()
with torch.no_grad():
    train_acc = evaluate_accuracy(train_iter, model)
    test_acc = evaluate_accuracy(test_iter, model)
outputs = model.trainModel(X_test.to("cuda"))
outputs = outputs.cpu().detach().numpy()
predict = np.vstack(outputs[:, 1])

threshold = 0.5
y_pred = (predict >= threshold).astype(int)
acc, auc, mcc, sensitivity, specificity = zhibiao(predict, y_pred, y_test)
PR = metrics.average_precision_score(y_test, predict)
