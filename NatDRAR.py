import tensorflow as tf
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import torchvision
from sklearn.model_selection import KFold,StratifiedKFold
from torch import optim
import random
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
BATCH_SIZE=128
Nc=10
Nf=10
N=Nc+Nf
N_max=100
EPOCH=1
best_fitness=95
min_fitness=80
omic_rate=0.1
random.seed(1)
gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lossFunction = nn.BCELoss()
class Dense5(nn.Module):
    def __init__(self):
        super(Dense5, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(2048, 500),
                                    nn.ReLU(True))
        self.dropout1=nn.Sequential( nn.BatchNorm1d(500),
                                        nn.Dropout(p=0.5))

        self.layer2 = nn.Sequential(nn.Linear(500, 200), nn.ReLU(True)
                                    )
        self.dropout2 = nn.Sequential(nn.BatchNorm1d(200),nn.Dropout(p=0.5))

        self.layer3 = nn.Sequential(nn.Linear(200, 100), nn.ReLU(True)
                                    )
        self.dropout3 = nn.Sequential(nn.BatchNorm1d(100),nn.Dropout(p=0.5))

        self.layer4 = nn.Sequential(nn.Linear(100, 30), nn.ReLU(True)
                                    )
        self.dropout4 = nn.Sequential(nn.BatchNorm1d(30),nn.Dropout(p=0.5))

        self.layer5 = nn.Sequential(nn.Linear(30, 10), nn.ReLU(True))

        self.layer6 = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x=self.dropout1(x)

        x = self.layer2(x)
        x=self.dropout2(x)

        x = self.layer3(x)
        x=self.dropout3(x)

        x = self.layer4(x)
        x = self.dropout4(x)

        x = self.layer5(x)
        x = self.layer6(x)
        return x
def init_net(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_()  # 全连接层参数初始化
            m.bias.data.normal_()
def get_threshold_metrics(y_true, y_pred, drop_intermediate=False,
                          disease='all'):
    """
    Retrieve true/false positive rates and auroc/aupr for class predictions

    Arguments:
    y_true - an array of gold standard mutation status
    y_pred - an array of predicted mutation status
    disease - a string that includes the corresponding TCGA study acronym

    Output:
    dict of AUROC, AUPR, pandas dataframes of ROC and PR data, and cancer-type
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import precision_recall_curve, average_precision_score

    roc_columns = ['fpr', 'tpr', 'threshold']
    pr_columns = ['precision', 'recall', 'threshold']

    if drop_intermediate:
        roc_items = zip(roc_columns,
                        roc_curve(y_true, y_pred, drop_intermediate=False))
    else:
        roc_items = zip(roc_columns, roc_curve(y_true, y_pred))

    roc_df = pd.DataFrame.from_dict(dict(roc_items))

    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    pr_df = pd.DataFrame.from_records([prec, rec]).T
    pr_df = pd.concat([pr_df, pd.Series(thresh)], ignore_index=True, axis=1)
    pr_df.columns = pr_columns

    auroc = roc_auc_score(y_true, y_pred, average='weighted')
    aupr = average_precision_score(y_true, y_pred, average='weighted')

    return {'auroc': auroc, 'aupr': aupr, 'roc_df': roc_df,
            'pr_df': pr_df, 'disease': disease}
def evalBEE(individual,loader):
     y_pred = list()
     y_true=list()
     net = Dense5().to(device)
     net = nn.DataParallel(net)
     net.load_state_dict(individual)
     net.eval()
     for j, data in enumerate(loader, 0):
         x, y = data
         x, y = x.to(device), y.to(device)
         outputs = net(x)
         _ = outputs.data.cpu().numpy()
         y=y.data.cpu().numpy()
         for i in range(len(_)):
             y_pred.append(_[i])
             y_true.append(y[i])
     return roc_auc_score(y_true, y_pred, average='weighted')*100
def predict(individual,loader):
    y_pred = list()
    y_true = list()
    net = Dense5().to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(individual)
    net.eval()
    for j, data in enumerate(loader, 0):
        x, y = data
        x, y = x.to(device), y.to(device)
        outputs = net(x)
        _ = outputs.data.cpu().numpy()
        y = y.data.cpu().numpy()
        for i in range(len(_)):
            y_pred.append(_[i])
            y_true.append(y[i])
    return  y_pred
def evolve(omic,individual1,individual2):
    for name in individual1:
      if 'dropout' not in name and 'weight' in name:
        individual1[name] = torch.tensor(
            individual1[name].cpu().numpy()+omic*(individual1[name].cpu().numpy()-individual2[name].cpu().numpy()),
            dtype=torch.float32).to(device)
    return individual1
def get_optim(net):
    p = random.uniform(0, 2.5)
    if p < 0.25:

        optimizers = optim.SGD(net.parameters(), lr=0.1)
    elif p < 0.5:
        optimizers = optim.Adam(net.parameters(), lr=0.01)
    elif p < 0.75:
        optimizers = optim.Adam(net.parameters(), lr=0.01)
    elif p < 1:
        optimizers = optim.RMSprop(net.parameters(), lr=0.01)
    elif p < 1.25:
        optimizers = optim.RMSprop(net.parameters(), lr=0.1)
    elif p < 1.5:

        optimizers = optim.Adadelta(net.parameters(), lr=0.1)
    elif p < 1.75:

        optimizers = optim.ASGD(net.parameters(), lr=0.1)
    elif p < 2:

        optimizers = optim.Rprop(net.parameters(), lr=0.1)
    elif p < 2.25:
        optimizers = optim.Adamax(net.parameters())
    elif p<2.5:

        optimizers = optim.SGD(net.parameters(), lr=0.01)

    return optimizers,p
def get_optim_by_p(net,p):
    if p < 0.25:
        optimizers = optim.SGD(net.parameters(), lr=0.1)
    elif p < 0.5:
        optimizers = optim.Adam(net.parameters(), lr=0.01)
    elif p < 0.75:
        optimizers = optim.Adam(net.parameters(), lr=0.01)
    elif p < 1:
        optimizers = optim.RMSprop(net.parameters(), lr=0.01)
    elif p < 1.25:
        optimizers = optim.RMSprop(net.parameters(), lr=0.1)
    elif p < 1.5:

        optimizers = optim.Adadelta(net.parameters(), lr=0.1)
    elif p < 1.75:

        optimizers = optim.ASGD(net.parameters(), lr=0.1)
    elif p < 2:

        optimizers = optim.Rprop(net.parameters(), lr=0.1)
    elif p < 2.25:
        optimizers = optim.Adamax(net.parameters())
    elif p < 2.5:

        optimizers = optim.SGD(net.parameters(), lr=0.01)

    return optimizers
def Bee(trainloader,valloader,bee=None):
    if bee==None:
        bee = list()
        for i in range(N):  # init
            net = Dense5().to(device)
            net = nn.DataParallel(net)
            init_net(net)
            bee.append(net.state_dict())
    elif len(bee)==Nc:
        for i in range(Nf):
            net = Dense5().to(device)
            net = nn.DataParallel(net)
            init_net(net)
            bee.append(net.state_dict())
    bee_fitness=list()
    for i in range(N):
        bee_fitness.append(evalBEE(bee[i],valloader))
    print(bee_fitness)
    for iter in range (N_max):
        print('iter:')
        print(iter)
        empolyee_bee = list()
        onlooker_bee = list()
        empolyee_bee_fitness = list()
        onlooker_bee_fitness = list()
        index = np.argsort(-np.array(bee_fitness))
        for i in range(N):
            if i < Nc:
                empolyee_bee.append(bee[index[i]])
                empolyee_bee_fitness.append(bee_fitness[index[i]])
            else:
                onlooker_bee.append(bee[index[i]])
                onlooker_bee_fitness.append(bee_fitness[index[i]])
        print('empolyee')
        print(empolyee_bee_fitness)
        print('worker')
        print(onlooker_bee_fitness)

        if empolyee_bee_fitness[0]>best_fitness:
            print('got best fitness')
            return empolyee_bee,empolyee_bee_fitness
        for i in range(Nc):
            if (i < Nc-1):
                temp_ind=evolve(omic_rate, empolyee_bee[i], empolyee_bee[i + 1])
            else:
                temp_ind=evolve(omic_rate, empolyee_bee[i], empolyee_bee[i - 1])
            net = Dense5().to(device)
            net = nn.DataParallel(net)
            net.load_state_dict(temp_ind)
            net.train()
            optimizers,p=get_optim(net)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode='min', factor=0.5, patience=5,
                                                             verbose=True, min_lr=1e-5, eps=0.0003)
            for epoch in range(EPOCH):
                train_loss=0
                for j, data in enumerate(trainloader, 0):
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    optimizers.zero_grad()
                    outputs = net(x)
                    loss = lossFunction(outputs, y)
                    loss.backward()
                    train_loss+=loss
                    optimizers.step()
                train_loss/=len(trainloader)
                scheduler.step(train_loss)
            _=evalBEE(net.state_dict(),valloader)
            if _>empolyee_bee_fitness[i]:
               print('adjust')
               while _>empolyee_bee_fitness[i]:
                   empolyee_bee_fitness[i] = _
                   empolyee_bee[i] = net.state_dict()
                   for epoch in range(EPOCH):
                       train_loss = 0
                       for j, data in enumerate(trainloader, 0):
                           x, y = data
                           x, y = x.to(device), y.to(device)
                           optimizers.zero_grad()
                           outputs = net(x)
                           loss = lossFunction(outputs, y)
                           loss.backward()
                           train_loss += loss
                           optimizers.step()
                       train_loss /= len(trainloader)
                       scheduler.step(train_loss)
                   _ = evalBEE(net.state_dict(), valloader)
        print('empolyee_bee_fitness')
        print(empolyee_bee_fitness)
        _=np.sum(np.array(empolyee_bee_fitness))
        probability=list()
        for k in range (Nc):
            probability.append(empolyee_bee_fitness[k]/_)
    
        for k in range (Nf):
            p_ = random.uniform(0, 1)
            q=0
            for kk in range (Nc):
                if p_<q+probability[kk]:
                    break
                else:
                    q=q+probability[kk]
            onlooker_bee[k]=empolyee_bee[kk]
            onlooker_bee_fitness[k]=evalBEE(onlooker_bee[k], valloader)
            net = Dense5().to(device)
            net = nn.DataParallel(net)
            net.load_state_dict(onlooker_bee[k])
            net.train()
            optimizer,p = get_optim(net)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                             verbose=True, min_lr=1e-5, eps=0.0003)
            for epoch in range(EPOCH):
                train_loss = 0
                for j, data in enumerate(trainloader, 0):
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = net(x)
                    loss = lossFunction(outputs, y)
                    loss.backward()
                    train_loss += loss
                    optimizer.step()
                train_loss /= len(trainloader)
                scheduler.step(train_loss)
            _ = evalBEE(net.state_dict(), valloader)
            print('worker fitenss')
            print(onlooker_bee_fitness[k])
            print(_)
            if _ > float(onlooker_bee_fitness[k]):
                print('adjust')
                while _ >  float(onlooker_bee_fitness[k]):
                    onlooker_bee_fitness[k] = _
                    onlooker_bee[k] = net.state_dict()
                    for epoch in range(EPOCH):
                        train_loss = 0
                        for j, data in enumerate(trainloader, 0):
                            x, y = data
                            x, y = x.to(device), y.to(device)
                            optimizer.zero_grad()
                            outputs = net(x)
                            loss = lossFunction(outputs, y)
                            loss.backward()
                            train_loss += loss
                            optimizer.step()
                        train_loss /= len(trainloader)
                        scheduler.step(train_loss)
                    _ = evalBEE(net.state_dict(), trainloader)
                onlooker_bee_fitness[k] = p
            onlooker_bee[k]= net.state_dict()
            onlooker_bee_fitness[k]=evalBEE(onlooker_bee[k],valloader)
        print('worker fitness')
        print(onlooker_bee_fitness)
        bee=empolyee_bee+onlooker_bee
        bee_fitness=empolyee_bee_fitness+onlooker_bee_fitness
    empolyee_bee = list()
    empolyee_bee_fitness = list()
    index = np.argsort(-np.array(bee_fitness))
    for i in range(N):
        if i < Nc:
            empolyee_bee.append(bee[index[i]])
            empolyee_bee_fitness.append(bee_fitness[index[i]])
    return empolyee_bee, empolyee_bee_fitness

print("Start Training, Dense5!")
print('reading')

x_df=pd.read_csv('genefile.csv',index_col=0,header=0)
y_df=pd.read_csv('statefile.csv',index_col=0,header=0)
strat=pd.read_csv('stratfile.csv',index_col=0,header=0)
print('read down')
x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(x_df.iloc[:,:2048], y_df, test_size=0.1, random_state=0,stratify=strat)
k_fold = StratifiedKFold(5,True, random_state=1)
index = k_fold.split(X=x_train_all, y=y_train_all)
cv=0
index_train_all=np.array(x_train_all.index)
cv_results_df=list()
for train_index, test_index in index:
        print('#########')
        x_train = np.array(x_train_all.iloc[train_index,:])
        x_val = np.array(x_train_all.iloc[test_index,:])
        y_train = np.array(y_train_all.iloc[train_index,:])
        y_val = np.array(y_train_all.iloc[test_index,:])
        x_train=torch.tensor(x_train,dtype=torch.float32)
        y_train=torch.tensor(y_train,dtype=torch.float32)
        x_val=torch.tensor(x_val,dtype=torch.float32)
        y_val=torch.tensor(y_val,dtype=torch.float32)
        dataset = TensorDataset(x_train, y_train)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        valdataset = TensorDataset(x_val,y_val)
        valloader = torch.utils.data.DataLoader(valdataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        if cv==0:
            bee,bee_fitness=Bee(trainloader,valloader)
        else:
            bee,bee_fitness = Bee(trainloader,valloader,bee)
        print ('best accuracy')
        print(bee_fitness[0])
        y_val_pred=predict(bee[0],valloader)
        cv_pred = np.concatenate((y_val_pred,y_val), axis=1)
        test_index = np.array(index_train_all[test_index])
        cv_df = pd.DataFrame(data=cv_pred, columns=['probability', 'real'], index=test_index)
        cv_results_df.append(cv_df)
        cv=cv+1


cv_df=pd.concat(cv_results_df,axis=0)
metrics_test=get_threshold_metrics(np.array(cv_df.loc[:,'total_status']),np.array(cv_df.loc[:,'dignosis']))
cv_df.to_csv('cv_results_NatDRAR.csv')


x_train = torch.tensor(np.array(x_train_all), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train_all), dtype=torch.float32)
dataset = TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

train_pred=np.concatenate((predict(bee[0],trainloader),np.array(y_train_all.loc[:,:])),axis=1)
train_df=pd.DataFrame(data=train_pred, columns=['dignosis','total_status'],index=x_train_all.index)
train_df.to_csv('train_results_NatDRAR.csv')

dataset = TensorDataset(torch.tensor(np.array(x_test_all),dtype=torch.float32),
                            torch.tensor(np.array(y_test_all),dtype=torch.float32))
testloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
y_pred=predict(bee[0],testloader)
test_pred=np.concatenate((y_pred,np.array(y_test_all.loc[:,:])),axis=1)
test_df=pd.DataFrame(data=test_pred, columns=['dignosis','total_status'],index=x_test_all.index)
test_df.to_csv('test_results_NatDRAR.csv')
torch.save(bee[0],'NatDRAR.pkl')