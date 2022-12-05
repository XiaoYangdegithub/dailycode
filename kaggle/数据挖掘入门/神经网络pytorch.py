# -*- coding:utf-8 -*-im
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.utils.data as D
import pandas as pd
import numpy as np
import copy
import os
from sklearn.metrics import roc_auc_score,log_loss
from tqdm.notebook import tqdm
from collections import defaultdict

#参数配置
config = {
    'train_path':"/kaggle/input/tabuar-playground-series-may-2022/train.csv",
    'test_path':'kaggle/input/tabuar-playground-series-may-2022/test.csv',
    'sparse_cols':['f_07','f_08','f_09','f_10','f_11','f_12','f_13','f_14','f_15','f_16','f_17'
,'f_18','f_29','f_30']+[f'ch_{i}' for i in range(10)],
    "dense_cols" : ['f_00','f_01','f_02','f_03','f_04','f_05','f_06','f_19','f_20','f_21','f_22'
,'f_23','f_24','f_25','f_26','f_28',],
    'debug_mode' : True,  #可以选一部分数据来跑，让模型跑的更快
    'epoch' : 5,
    'batch' : 2048,
    'lr' : 0.001,  #设置学习率
    'device' : 0 #GPU选择
} #用字典的形式配置参数

train_df = pd.read_csv(config['train_path'])
if config['debug_mode']:
    train_df = train_df[:1000]
test_df = pd.read_csv(config['test_path'])

df = pd.concat([train_df, test_df],axis=0).rest_index(drop=True)  #利用pd.concat合并（上下合并）数据集并重置参数
for i in tqdm(range(10)):
    df[f'ch_{i}'] = df['f_27'].str.get(i).apply(ord) - ord('A')

def get_enc_dict(df,config): #计算enc_dict 离散特征不能直接编码放在模型里
    enc_dict = defaultdict(dict)
    for f in tqdm(config['sparse_cols']): #离散特征
        map_dict = dict(zip(df[f].unique(), range(1, df[f].nunique() + 1))) #从1的位置开始编码，0位置留给位置数据
        enc_dict[f] = map_dict
        enc_dict[f]["vocab_size"] = df[f].unique() + 1

    for f in tqdm(config['dense_cols']): #连续特征
        enc_dict[f]['min'] = df[f].min()
        enc_dict[f]['max'] = df[f].max()
        enc_dict[f]['std'] = df[f].std()

    return enc_dict
enc_dict = get_enc_dict(df,config)

#构建Dataset
class BaseDataset(Dataset):
    def __init__(self,config,df,enc_dict=None):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict #赋值
        self.dense_cols = list(set(self.config['dense_cols'])) # 连续特征导入，set把导入的数据变为集合
        # --为了去除重复数据， 再初始化为一个list
        self.sparse_cols = list(set(self.config['sparse_cols']))
        self.feature_name = self.dense_cols+self.sparse_cols+['label']
        self.enc_data()
    # 数据编码 在Dataset中编码
    def enc_dense_data(self,col):
        return (self.df[col] - self.enc_dict[col]['min'] / (self.enc_dict[col]['max'] -
self.enc_dict[col]['min']))  # min-max方法处理连续特征

    def enc_sparse_data(self,col): # 使用apply()方法处理离散特征
        return self.df[col].apply(lambda x :self.enc_dict[col].get(x,0))
    # 声明一个lambda表达式，x是表达式中输入的参数，self.enc_dict查询字典，字典中若有x则返回对应的key，没有的话返回0

    def enc_data(self): #使用enc_dict对数据进行编码
        self.enc_df = copy.deepcopy(self.df)
        for col in self.dense_cols:
            self.enc_df[col] = self.enc_dense_data(col)
        for col in self.sparse_cols:
            self.enc_df[col] = self.enc_sparse_data(col)

    def __getitem__(self, index): #返回第index个数据
        data = dict() # 针对特征个数不确定的情况，使用字典格式
        for col in self.feature_name: # 对离散特征和连续特征进行赋值，赋值到字典里面
            if col in self.dense_cols:
                data[col] = torch.Tensor([self.enc_df[col].iloc[index]]).squeeze(-1) # 把张量降一维
            elif col in self.sparse_cols:
                data[col] = torch.Tensor([self.enc_df[col].iloc[index]]).long().squeeze(-1)
                # 把离散特征转为long()才能再转为字典
        if 'target' in self.enc_df.columns:
            data['target'] = torch.Tensor([self.enc_df['target'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.enc_df) # 数据集的长度

train_df = df[df['target'].notna()].reset_index(drop=True)
test_df = df[df['target'].isna()].reset.index(drop=True)

train_num = int(len(train_df)*0.8)
valid_df = train_df[train_num].reset_index(drop=True)
train_df = train_df[:train_num].reset_index(drop=True) #划分验证集和训练集

trian_dataset = BaseDataset(config,train_df,enc_dict=enc_dict)
valid_dataset = BaseDataset(config,valid_df,enc_dict=enc_dict)
test_df = BaseDataset(config,test_df,enc_dict=enc_dict)
trian_dataset.__getitem__(5) #查看index=5位置上的数据的特征和数据组成的字典
# 定义模型
emb_layer = nn.Embedding(8,4) # embedding中有8个词/embedding的词长度为8  将每个词编码为4维向量
emb_layer.weight
idx = torch.from_numpy(np.array([1,2,1])) #选取emb的第一、二、一的数据
emb_layer(idx)
#基本网络模块
#通用Emb 手动封装一个支持多个emb的emb层 ---把多个整合成一个大层
class Embeddinglayer(nn.Model):
    def __init__(self,
                 enc_dict = None,
                 embedding_dim = None):
        super(Embeddinglayer, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModelDict() # 使用一个字典

        self.emb_feature = []

        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col : nn.Embedding(
                    self.enc_dict[col]['vocab_size'],
                    self.embedding_dim,
                )}) # 再更新字典，key是embedding
    def forward(self, X): # 对所有的sparse特征逐个进行embedding
        feature_emb_list = []
        for col in self.emb_feature:#遍历所有的离散特征
            # 因为模型处理的时候是一个batch一个batch进行的，所以把离散特征归为一组一组的batch
            inp = X[col].long().view(-1, 1)
            # 把离散特征转变为[batch,1]第一项batch中的数据量，第二项为离散特征的个数
            feature_emb_list.append(self.embedding_layer[col](inp))#转变为[batch,embedding_dim]
        feature_emb = torch.stack(feature_emb_list, dim = 1)#堆叠为[batch,num_sparse_feature,embedding_dim]
        return feature_emb

# DNN
class MLP_Layer(nn.Model):
    def __init__(self,
                 input_dim,
                 output_dim=None,
                 hidden_units=[],  # 隐藏节点
                 hidden_activations='ReLU', # 激活函数
                 final_activation=None,
                 dropout_rates=0,
                 batch_norm=False,
                 use_bias=True):
        super(MLP_Layer,self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [set_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1],bias = use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if final_activation is not None:
            dense_layers.append(set_activation(final_activation))
        self.dnn = nn.Sequential(*dense_layers)

    def forward(self, inputs):
        return self.dnn(inputs)

def set_device(gpu=-1):
    if gpu >=0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICE'] = str(gpu)
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')
        return device

def set_activation(activation): # 激活函数
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation

def get_dnn_input_dim(enc_dict,embedding_dim): # 获取dnn的输入维度
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys(): # 有min就是连续特征
            num_dense +=1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse +=1
    return num_sparse*embedding_dim+num_dense

def get_linear_input(enc_dict, data): #获取连续特征输入
    res_data =[]
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            res_data.append(data[col])
    res_data = torch.stack(res_data,axis=1)
    return res_data

class TPSModle(nn.Model):
    def __init__(self,
                 embedding_dim = 16,
                 hidden_units = [64,32,16],
                 enc_dict = None,
                 hidden_activations = 'relu',
                 dropout_rates = 0,
                 loss_fun = 'torch.nn.BCELoss()'):
        super(TPSModle, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.enc_dict = enc_dict
        self.hidden_activations = hidden_activations
        self.dropout_rates = dropout_rates
        self.loss_fun = eval(loss_fun)  #把字符串执行一遍=torch.nn.BCELoss()

        self.embedding_layer = Embeddinglayer(enc_dict=self.enc_dict,
                                              embedding_dim=self.embedding_dim)

        self.dnn_input_dim = get_dnn_input_dim(enc_dict=self.enc_dict,
                                               embedding_dim=self.embedding_dim)
        # num_dense + num_sparse * embedding_dim
        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim,
                             out_put=1,
                             hidden_units=self.hidden_units,
                             hidden_activations=self.hidden_activations,
                             dropout_rates = self.dropout_rates)

    def forward(self,data):
        sparse_embedding = self.embedding_layer(data)  #batch,num_sparse,embedding-dim
        sparse_embedding = torch.flatten(sparse_embedding, start_dim = 1) # batch,num_sparse * embedding-dim
        dense_input = get_linear_input(enc_dict=self.enc_dict, data=data) #batch,num_sparse
        dnn_input = torch.cat([sparse_embedding, dense_input],axis = 1)#batch,num_sparse + num_sparse * embedding-dim
        y_pred = self.dnn(dnn_input).sigmoid()
        loss = self.loss_fun(y_pred.squeeze(-1),data['target'])
        output_dict = {'pred':y_pred,'loss':loss}
        return output_dict

# 训练验证模型
def train_model(model, train_loader, optimizer, device, metric_list=['roc_auc_score','log_loss']):
    model.train()
    pred_list = []
    label_list = []
    pbar = tqdm(train_loader)
    for data in pbar:
        for key in data.keys():
            data[key] = data[key].to(device)

        output = model(data)
        pred = output['pred']
        loss = output['loss']

        loss.backward()
        optimizer.step()
        model.zero_grad()

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['target'].squeeze(-1).cpu().detach().numpy())
        pbar.set_description('Loss{}'.format(loss))

    res_dict = dict() # 用字典保存指标的值
    for metric in metric_list:
        if metric == 'log_loss':
            res_dict[metric] = log_loss(label_list,pred_list, eps=1e-7)
        else:
            res_dict[metric] = eval(metric)(label_list,pred_list)

    return res_dict

def valid_model(model, valid_lodar, device, metric_list=['roc_auc_scroe','log_loss']):
    model.eval()
    pred_list = []
    label_list = []

    for data in (valid_lodar):
        for key in data.keys():
            data[key] = data[key].to(device)

        output = model(data)
        pred = output['pred']

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['target'].squeeze(-1).cpu().detach().numpy())

    res_dict = dict()
    for metric in metric_list:
        if metric == 'log_loss':
            res_dict[metric] = log_loss(label_list,pred_list, eps=1e-7)
        else:
            res_dict[metric] = eval(metric)(label_list, pred_list)
        return res_dict

def test_model(model, test_lodar, device):
    model.eval()
    pred_list = []

    for data in tqdm(test_lodar):

        for key in data.keys():
            data[key] = data[key].to(device)

        output = model(data)
        pred = output['pred']
        pred_list.extend(pred.squeeze().cpu().detach().numpy())

    return np.array(pred_list)
# dataloader
train_loader = D.DataLoader(trian_dataset,batch_size=config['batch'],shuffle=True,num_workers=0)
valid_loader = D.DataLoader(valid_dataset,batch_size=config['batch'],shuffle=False,num_workers=0)
test_loader = D.DataLoader(test_dataset,batch_size=config['batch'],shuffle=False,num_workers=0)
#shuffle 决定数据是否要打乱 一般训练数据要打乱 但是测试和验证数据要按顺序开始。num_workers代表多任务个数（多线程）多线程可减少IO时间

model = TPSModle(enc_dict=enc_dict)
device = set_device(config["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
model = model.to(device)
# 模型训练流程
for i in range(config['epoch']):
    trian_metric = train_model(model,train_loader,optimizer=optimizer,device=device)# 模型训练
    valid_metric = valid_model(model,valid_loader,device) # 模型验证

    print('Train Metric:')
    print(trian_metric)
    print('Valid Metric:')
    print(valid_metric)

y_pre = test_model(model, test_loader, device)

res_df = pd.DataFrame()
res_df['id'] = test_df['id']
res_df['target'] = y_pre
res_df.to_csv('/kaggle/working/torch_baseline.csv',index=False)













