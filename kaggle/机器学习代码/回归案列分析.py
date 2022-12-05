# -*- coding:utf-8 -*-
# 丰田卡罗拉价格回归分析案例
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from collections import Counter
from IPython.core.display import display, HTML
import warnings
warnings.filterwarnings('ignore')  # 忽略警告
dataset = pd.read_csv('data/ToyotaCorolla.csv')
'''
print(dataset.head())
print(dataset.count)
print(dataset.describe()) # 通过描述性统计分析数据
包含数据记录数、平均值、标准方差、最小值、下四分位数、中位数、上四分位数、最大值
'''
# 数据处理和可视化
dataset.isnull().sum()
corr = dataset.corr() # 理解数据属性间的相关性，1 表示变量完全正相关，0 表示无关，-1 表示完全负相关
# 当相关性比较高的时候就要考虑降维
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr, cmap='magma', annot=True, fmt='.2f')  # 生成热力图
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

f, axes = plt.subplots(2, 2, figsize=(12, 8)) # 建立一个fig对象和axis对象，绘制2*2个子图 figsize设置对象大小
sns.regplot(x='Price',
            y='Age',
            data=dataset,
            scatter_kws={'alpha': 0.6},
            ax=axes[0, 0])  # 绘制线性回归图
axes[0, 0].set_xlabel('Price', fontsize=14)
axes[0, 0].set_ylabel('Age', fontsize=14)
axes[0, 0].yaxis.tick_left()

sns.regplot(x='Price',
            y='KM',
            data=dataset,
            scatter_kws={'alpha': 0.6},
            ax=axes[0, 1])
axes[0, 1].set_xlabel('Price', fontsize=14)
axes[0, 1].set_ylabel('KM', fontsize=14)
axes[0, 1].yaxis.set_label_position('right')
axes[0, 1].yaxis.tick_right()

sns.regplot(x='Price',
            y= 'Weight',
            data=dataset,
            scatter_kws={'alpha': 0.6},
            ax=axes[1, 0])
axes[1, 0].set_xlabel('Price', fontsize=14)
axes[1, 0].set_ylabel('Price', fontsize=14)

sns.regplot(x='Price',
            y='HP',
            data=dataset,
            scatter_kws={'alpha': 0.6},
            ax=axes[1, 1])
axes[1, 1].set_xlabel('Price', fontsize=14)
axes[1, 1].set_ylabel('HP', fontsize=14)
axes[1, 1].yaxis.set_label_position('right')
axes[1, 1].yaxis.tick_right()
axes[1, 1].set(ylim=(40, 160))

plt.show()

f, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.displot(dataset['KM'], ax = axes[0])
axes[0].set_xlabel('KM', fontsize=14)
axes[0].set_ylabel('Count', fontsize=14)
axes[0].yaxis.tick_left()

sns.scatterplot(x='Price', y='KM', data=dataset, ax=axes[1])
axes[1].set_xlabel('Price', fontsize=14)
axes[1].set_ylabel('KM', fontsize=14)
axes[1].yaxis.set_label_position('right')
axes[1].yaxis.tick_right()
plt.show()
'''
fuel_list = Counter(dataset['FuelType'])
labels = fuel_list.keys()
sizes = fuel_list.values()

f, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.countplot(dataset['FuelType'], ax=axes[0], palette='Set1')
axes[0].set_xlabel('Fuel Type', fontsize=14)
axes[0].set_ylabel('Price', fontsize=14)
axes[0].yaxis.tick_left()

sns.violinplot(x='FuelType', y='Price', data=dataset, ax=axes[1])
axes[1].set_xlabel('Fuel Type', fontsize=14)
axes[1].set_ylabel('Price', fontsize=14)
axes[1].yaxis.set_label_position('right')
axes[1].yaxis.tick_right()

f, axes = plt.subplots(1,2,figsize=(14,4))

sns.distplot(dataset['HP'], ax = axes[0])
axes[0].set_xlabel('HP', fontsize=14)
axes[0].set_ylabel('Count', fontsize=14)
axes[0].yaxis.tick_left()

sns.scatterplot(x = 'HP', y = 'Price', data = dataset, ax = axes[1])
axes[1].set_xlabel('HP', fontsize=14)
axes[1].set_ylabel('Price', fontsize=14)
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()

plt.show()

f, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.distplot(dataset['MetColor'], ax=axes[0])
axes[0].set_xlabel('MetColor', fontsize=14)
axes[0].set_ylabel('Count', fontsize=14)
axes[0].yaxis.tick_left()

sns.boxplot(x='MetColor', y='Price', data=dataset, ax=axes[1])
axes[1].set_xlabel('MetColor', fontsize=14)
axes[1].set_ylabel('Price', fontsize=14)
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()

plt.show()

f, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.distplot(dataset['Automatic'], ax=axes[0])
axes[0].set_xlabel('Automatic', fontsize=14)
axes[0].set_ylabel('Count', fontsize=14)
axes[0].yaxis.tick_left()

sns.boxenplot(x='Automatic', y='Price', data=dataset, ax=axes[1])
axes[1].set_xlabel('Automatic', fontsize=14)
axes[1].set_ylabel('Price', fontsize=14)
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()

plt.show()

f, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.distplot(dataset['CC'], ax=axes[0])
axes[0].set_xlabel('CC', fontsize=14)
axes[0].set_ylabel('Count', fontsize=14)
axes[0].yaxis.tick_left()

sns.boxplot(x='CC', y='Price', data=dataset, ax=axes[1])
axes[1].set_xlabel('CC', fontsize=14)
axes[1].set_ylabel('Price', fontsize=14)
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()

plt.show()

f, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.distplot(dataset['Doors'], ax=axes[0])
axes[0].set_xlabel('Doors', fontsize=14)
axes[0].set_ylabel('Count', fontsize=14)
axes[0].yaxis.tick_left()

sns.boxenplot(x='Doors', y='Price', data=dataset, ax=axes[1])
axes[1].set_xlabel('Doors', fontsize=14)
axes[1].set_ylabel('Price', fontsize=14)
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()

plt.show()
'''
dataset = pd.get_dummies(dataset)  # 对特征进行编码 -- 利用pandas进行one hot encode
print(dataset.head())

x = dataset.drop('Price', axis=1).values
print(x)
y = dataset.iloc[:, 0].values.reshape(-1, 1)
print(y)
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,  # 所要划分的样本特征集
                                                    y,  # 所要划分的样本结果
                                                    test_size=0.25,  # 样本占比
                                                    random_state=42)  # 随机数种子
print('Shape of X_train:', X_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of y_test:', y_test.shape)

# 回归模型
# 线性回归
from sklearn.linear_model import LinearRegression
regressor_linear = LinearRegression()
regressor_linear.fit(X_train, y_train)  # 使用最小二乘法对训练集进行拟合
from sklearn.metrics import r2_score  # 回归模型的评价指标
cv_linear = cross_val_score(estimator=regressor_linear,  # 模型
                            X=X_train,
                            y=y_train,
                            cv=10)  # 交叉验证生成器可迭代的次数
# 训练集结果的R2分数，决定系数
y_pred_linear_train = regressor_linear.predict(X_train)
r2_score_linear_train = r2_score(y_train, y_pred_linear_train)
# 测试集结果的R2分数
y_pred_linear_test = regressor_linear.predict(X_test)
r2_score_linear_test = r2_score(y_test, y_pred_linear_test)
# 测试集结果的RMSE -- 均方根误差，一般不用来评判拟合
rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_test)))
print('CV:', cv_linear.mean())
print('R2_score(train):', r2_score_linear_train)
print('R2_score(test):', r2_score_linear_test)
print('RMSE:', rmse_linear)

# 二项多项式回归
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)  # 如果一元线性回归效果不佳，使用多元线性回归
X_poly = poly_reg.fit_transform(X_train)  # 进行特征处理，找出x_train的均值和方差，并应用到x_train上
poly_reg.fit(X_poly, y_train)
regressor_poly2 = LinearRegression()
regressor_poly2.fit(X_poly, y_train)

from sklearn.metrics import r2_score
cv_poly2 = cross_val_score(estimator=regressor_poly2, X=X_train, y=y_train, cv=10)
y_pred_poly2_train = regressor_poly2.predict(poly_reg.fit_transform(X_train))
r2_score_ploy2_train = r2_score(y_train, y_pred_linear_train)
y_pred_poly2_test = regressor_poly2.predict(poly_reg.fit_transform(X_test))
r2_score_ploy2_test = r2_score(y_test, y_pred_poly2_test)
rmse_ploy2 = (np.sqrt(mean_squared_error(y_test, y_pred_poly2_test)))
print('CV:', cv_poly2.mean())
print('R2_score(train):', r2_score_ploy2_train)
print('R2_score(test):', r2_score_ploy2_test)
print('RMSE', rmse_ploy2)

# 岭回归 -- 使用了正则化项解决过拟合问题，适用于特征间相关性高的问题
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

steps = [
    ('scalar', StandardScaler()),  # 数据标准化处理
    ('poly', PolynomialFeatures(degree=3)),
    ('model', Ridge(alpha=1777, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)

from sklearn.metrics import r2_score
# 在测试集上预测交叉验证分数
cv_ridge = cross_val_score(estimator=ridge_pipe, X=X_train, y=y_train.ravel(), cv=10)

# 在测试集上预测R2分数
y_pred_ridge_train = ridge_pipe.predict(X_train)
r2_score_ridge_train = r2_score(y_train, y_pred_ridge_train)
y_pred_ridge_test = ridge_pipe.predict(X_test)
r2_score_ridge_test = r2_score(y_test, y_pred_ridge_test)

# 预测测试集上的RMSE
rmse_ridge = (np.sqrt(mean_squared_error(y_test, y_pred_ridge_test)))
print('CV:', cv_ridge.mean())
print('R2_score(train):', r2_score_ridge_train)
print('R2_score(test):', r2_score_ridge_test)
print('RMSE: ', rmse_ridge)

# 套索回归
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

steps = [('scalar', StandardScaler()), ('poly', PolynomialFeatures(degree=3)),
         ('model',
          Lasso(alpha=2.36, fit_intercept=True, tol=0.0199, max_iter=2000))]
# alpha-与L1项相乘的系数 fit_intercept-是否计算模型截距 tol-优化的容限 max_iter-最大迭代次数

lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)

from sklearn.metrics import r2_score

cv_lasso = cross_val_score(estimator=lasso_pipe, X=X_train, y=y_train, cv=10)
y_pred_lasso_train = lasso_pipe.predict(X_train)
r2_score_lasso_train = r2_score(y_train, y_pred_lasso_train)
y_pred_lasso_test = lasso_pipe.predict(X_test)
r2_score_lasso_test = r2_score(y_test, y_pred_lasso_test)
rmse_lasso = (np.sqrt(mean_squared_error(y_test, y_pred_lasso_test)))
print('CV：', cv_lasso.mean())
print('R2_score(train):', r2_score_lasso_train)
print('R2_score(test):', r2_score_lasso_test)
print('RMSE:', rmse_lasso)

# 误差衡量
models = [
    ('Linear Regression', rmse_linear, r2_score_linear_train,
     r2_score_linear_test, cv_linear.mean()),
    ('Polynomial Regression(2nd)', rmse_ploy2, r2_score_ploy2_train,
     r2_score_ploy2_test, cv_poly2.mean()),
    ('Ridge Regression', rmse_ridge, r2_score_ridge_train,
     r2_score_ridge_test, cv_ridge.mean()),
    ('Lasso Regression', rmse_lasso, r2_score_lasso_train, r2_score_lasso_test,
     cv_lasso.mean()),
]

predict = pd.DataFrame(data=models, columns=['Model', 'RMSE', 'R2_Score(training)', 'R2_Score(test)', 'Cross-Validation'])
print(predict)
# 模型性能可视化

# 交叉验证集
f, axe = plt.subplots(1, 1, figsize=(18, 6))

predict.sort_values(by=['Cross-Validation'], ascending=False, inplace=True)

sns.barplot(x='Cross-Validation', y='Model', data = predict, ax = axe)
axe.set_xlabel('Cross-Validation Score', size=16)
axe.set_ylabel('Model', size=16)
axe.set_xlim(1, 1.0)
axe.set_xticks(np.arange(0, 1.1, 0.1))
plt.show()

# R2
f, axes = plt.subplots(2, 1, figesize=(14, 10))

predict.sort_values(by=['R2_Score(training)'], ascending=False, inplace=True)

sns.barplot(x='R2_Score(training)',
            y='Model',
            data=predict,
            palette='Blues_d',
            ax=axes[0])
axes[0].set_xlabel('R2 Score (Training)', size=16)
axes[0].set_ylabel('Model', size=16)
axes[0].set_xlim(0, 1.0)
axes[0].set_xticks(np.arange(0, 1.1, 0.1))
predict.sort_values(by=['R2_Score(test)'], ascending=False, inplace=True)
sns.barplot(x='R2_Score(test)',
            y='Model',
            data=predict,
            palette='Reds_d',
            ax=axes[1])
axes[1].set_xlabel('R2 Score (Test)', size=16)
axes[1].set_ylabel('Model', size=16)
axes.set_xlim(0, 1.0)
axes.set_xticks(np.arange(0, 1.1, 0.1))
plt.show()

# RMSE
predict.sort_values(by=['RMSE'], ascending=False, inplace=True)
f, axe = plt.subplots(1, 1, figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=predict, ax=axe)
axe.set_xlabel('Model', size=16)
axe.set_ylabel('RMSE', size=16)
plt.show()

