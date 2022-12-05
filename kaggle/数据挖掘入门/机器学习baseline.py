# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os
for dirname, _ ,filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname.filename))

from tqdm.notebook import tqdm #进度条展示
from matplotlib import pyplot as plt
import seaborn as sns #画图
import lightgbm as lgb #后续模型展示

#from sklearn.preprocessing import LabelEncoder, MinMaxScler, StandarScaler #导入sklearn库
#from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFlod
from sklearn.metrics import roc_auc_score, log_loss

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')
df = pd.concat([train_df, test_df],axis=0).reset_index(drop=True) #将训练数据集与测试数据集合并 更新参数
train_df.head() #查看头部数据集
train_df.tail() #查看尾部数据集
df.info(verbose=True, null_counts=True) #是否打印log，是否统计缺失值个数
df['f_27'] #因为27是一个特殊的离散值，要进行处理后编码
for i in tqdm(range(10)) :
    df[f'ch_{i}'] = df['f_27'].str.get(i).apply(ord) - ord('A')
#apply(ord)转化为大写，- ord('A')显示距离A的距离将27的模型特征共十个划分为单独的十个特征分别判断

num_cols  =  ['f_00',
             'f_01',
             'f_02',
             'f_03',
             'f_04',
             'f_05',
             'f_06',
             'f_19',
             'f_20',
             'f_21',
             'f_22',
             'f_23',
             'f_24',
             'f_25',
             'f_26',
             'f_28',]  #连续特征
cate_cols = ['f_07',
             'f_08',
             'f_09',
             'f_10',
             'f_11',
             'f_12',
             'f_13',
             'f_14',
             'f_15',
             'f_16',
             'f_17',
             'f_18',
#              'f_27', #因为27已经特殊处理 所以注释掉
             'f_29',
             'f_30'] + [f'ch_{i}' for i in range(10)]  # 离散特征
#连续特征图像展示和标准化
for col in tqdm(num_cols):
    plt.figure(dpi=150) # 把图放大，dpi表示清晰度
    sns.displot(df[col]) # 将连续特征用图表示出来

for col in tqdm(num_cols):
    # min-max标准化：对原始数据进行线性变换，将值映射到[0，1]之间
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    # z-score标准化：将不同量级的数据统一化为同一个量级
    df[col] = (df[col] - df[col].mean()) / df[col].std()

#sklearn数据编码
# scaler = StandardScaler() # z-score直接调用
# scaler = MinMaxScaler() #min-max直接调用
# df[num_cols] = scaler.fit_transform(df[num_cols]) # 一次可处理多个特征

df.describe() # 数据标准化后的展现
# 离散特征图像展示与标准化
for col in tqdm(cate_cols):
    plt.figure(dpi=150)
    sns.countplot(df[col]) #离散特征频次分布的展现

for col in tqdm(cate_cols):
    map_dict = dict(zip(df[col].unique(), range(df[col].nunique())))
# unique（）可以将离散数据中的重复数据删除；nunique（）表示unique的长度
    df[col] = df[col].map(map_dict) # 把离散特征的值换成dict里的对应的key
    df[f'{col}_count'] = df[col].map(df[col].value_counts()) # 每一个离散特征出现的频次

# sklearn 离散特征的编码
#for col in tqdm(cate_cols):
#   scale = LabelEncoder()
#   scale.fit(df[col])
#   df[col] = scale.transform(df[col])

# 划分训练集和测试集：训练集的target为非空、测试集的target为空
train_df = df[df['target'].notna()].reset_index(drop=True)
test_df = df[df['target'].isna()].reset_index(drop=True)
drop_feature = ['id','target', 'f_27'] #将无用的特征丢掉
feature = [x for x in train_df.columns if x not in drop_feature]
print(len(feature),feature)
#模型调用
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_df[feature], train_df['target'], random_state=666)

from sklearn.linear_model import LinearRegression # 线性回归
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB # 朴素贝叶斯
from sklearn.svm import SVC #svm
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn.neural_network import MLPClassifier #人工神经网络
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier #提升算法

model = LogisticRegression() #可以使用以上的任何一种模型
model.fit(X_train, y_train)
y_valid_pre = model.predict_proba(X_valid)[:,1] #预测类别为1的概率
print(f'{str(model)}AUC :{roc_auc_score(y_valid, y_valid_pre)}') #第一项为标签，第二项为概率
print(f'{str(model)}logloss :{log_loss(y_valid, y_valid_pre)}') #loss
y_pre = model.predict_proba(test_df[feature])[:,1]

model.predict_proba(X_valid)
#模型验证
params = {'num_leaves': 60, #结果对最终结果影响大，越大越好，但太大易出现过拟合
          'min_data_in_leaf':30,
          'objective': 'binary', #定义目标函数
          'max_depth': -1,
          'learning_rate': 0.1,
          'min_sum_hessian_in_leaf': 6,
          'boosting': 'gbdt',
          'feature_fraction': 0.9, #提取的特征比率
          'bagging_freq': 1,
          'bagging_fraction': 0.8,
          'bagging_seed':11,
          'lambda_11':0.1,  #l1正则化
          #'lambda_12‘：0.001, #l2正则化
          'verbosity': -1,
          'nthread': -1, #线
          # 程数量，-1代表全部线程，线程越多，运行速度越快
          'metric':{'binary_logloss','auc'}, #评价函数
          'random_state':2019 #随机种子，防止每次运行结果不一致
          #'device':'gpu' #如果安装代理如GPU，则会加快运算

}# 调参 一般调num_leaves和learning_rate
#交叉验证 - 选择k折交叉验证法
n_flod = 5
oof_pre = np.zeros(len(train_df))
y_pre = np.zeros(len(test_df))

kf = KFold(n_splits=n_flod)
for fold_,(trn_idx, val_idx) in enumerate(kf.split(train_df)):
    trn_data = lgb.Dataset(train_df[feature].lioc[trn_idx], label=train_df['target'].lioc[trn_idx])
    val_data = lgb.Dataset(train_df[feature].iloc[val_idx], label=train_df['target'].lioc[val_idx])

    clf = lgb.train(params,
                    trn_data,
                    10000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=50,
                    early_stopping_rounds=50)
    oof_pre[val_idx] = clf.predict(train_df[feature].iloc[val_idx], num_iteration=clf.best_iteration)
    #oof_pre指交叉验证之后得到的对训练集的预测
    y_pre += clf.predict(test_df[feature], num_iteration=clf.best_iteration) / n_flod #获得最后的验证预测值
rest_df = pd.DataFrame()
rest_df['id'] = test_df['id']
rest_df['target'] = y_pre
rest_df.to_csv('/kaggle/working/baseline.csv',index=False)

#使用HPO来调整超参数训练模型
from openbox import sp.Optimizer #导入openbox
import  warnings #忽略一些没有必要的警告
warnings.filterwarnings('ignore')

HPO_DEBG = False
#HPO处理
from sklearn.model_selection import train_test_split
if HPO_DEBG: #true的情况下使用10000条数据来跑，验证模型是否运行正常
    X_train, X_valid, y_train, y_valid = train_test_split(train_df[feature][:10000], train_df['taget'][:10000], random_state=666)
else: #flase的情况下是将所有数据跑
    X_train, X_valid, y_train, y_valid = train_test_split(train_df[feature], train_df['target'], random_state=666)

def get_configspace(): #定义搜索空间
    space = sp.Space()
    n_estimators = sp.Int('n_estimators', 100, 1000, default_value=500, q=50)#迭代次数
    num_leaves = sp.Int('num_leaves', 31, 2047, default_value =128)#叶子个数
    max_depth = sp.Int('max_depth', 15, 127, default_value=31, q=16)  # 最大深度
    learning_rate = sp.Real('learning_rate', 1e-3, 0.3, default_value=0.1, log=true)#学习率
    min_child_samples = sp.Int('min_child_samples', 5, 30, default_value=20)
    subsample = sp.Real('subsample', 0.7, 1,default_value=1, q=0.1)
    colsample_bytree = sp.Real('colsample_bytree', 0.7, 1,default_value=1, q=0.1)

    space.add_variable([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                        colsample_bytree]) #相加得到最终的搜索空间
    return space
space = get_configspace()

def objective_function(config: sp.Configuration): #定义目标函数
    params = config.get_dictionary() #获取到参数的字典

    params['n_jobs'] = 2
    params['random_state'] = 47 #保证可以复现
    params['verbose'] = -1 #模型不输出日志

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pre = model.predict_proba(X_valid)[:,1]
    auc = roc_auc_score((y_valid, y_pre))

    return dict(objs=(-auc,))#因为openbox里的准则是越小越好，需要加一个负号

opt = Optimizer(
    objective_function(),
    get_configspace(),
    num_objs=1,
    num_constrains=0,
    max_runs=20, #搜索的轮次
    surrogate_type='gp', #代理模型-高斯回归
    time_limit_per_trial=18000, #每次搜索的最大耗时
    task_id='so_hpo',
)
history = opt.run() #开始超参优化
all_prefs = history.get_all_prefs() #拿出所有模型的指标
best_index = np.argmax(-np.array(all_prefs)) #argmax取到对应最大模型的指标
best_params = history.get_all_configs()[best_index] #把最大指标对应的超参拿出
print(best_params)
history.plot_convergence() #展示超参调节的过程
history.visualize_jupyter() #超参的结果

#继续利用n折验证


