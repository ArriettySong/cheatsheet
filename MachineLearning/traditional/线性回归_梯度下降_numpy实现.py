import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

class LinearRegression_GradientDescent:

    def __init__(self,alpha,times):
        """
        alpha: 学习率
        times：迭代次数
        """
        self.alpha = alpha
        self.times = times

    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.w_ = np.zeros(X.shape[1] +1 ) #加w0
        self.loss = []

        # loss计算公式：(y-y_hat)的平方和的一半
        # 权重w0调整公式：w0 = w0 + alpha * sum(y-y_hat)
        # 权重wi调整公式：w(j) = w(j) + alpha * sum((y-y_hat) * x(j))

        for i in range(1,self.times):
            y_hat = np.dot(X,self.w_[1:]) + self.w_[0]
            error = y - y_hat
            self.loss.append(np.sum(error ** 2)/2)
            self.w_[0] += self.alpha * np.sum(error)
            self.w_[1:] += self.alpha * np.dot(X.T,error)


    def predict(self,X):
        X = np.asarray(X)
        result = np.dot(X,self.w_[1:]) + self.w_[0]
        return result


# 原始的梯度下降
boston = datasets.load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names
print("feature_names:",feature_names)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print("train data's shape:",X_train.shape)
print("test data's shape:",X_test.shape)

lr = LinearRegression_GradientDescent(alpha=0.0005,times=10)
lr.fit(X_train,y_train)
y_predict = lr.predict(X_test)
print("y_predict shape:",y_predict.shape)
print("MSE:", np.mean(((y_predict-y_test)**2)) ) # 求均方误差MSE
print("求解得到的w:",lr.w_)
print("loss变化情况:",lr.loss)

class StandardScaler:
    '''
    对数据集进行标准化处理
    '''
    def fit(self,X):
        """
        基于训练数据计算每一列的均值和标准差
        """
        X = np.asarray(X)
        self.mean_ = np.mean( X,axis=0 )# 按列计算标准差
        self.std_ = np.std( X,axis=0 )# 按列计算均值

    def transform(self,X):
        """
        对给定数据进行标准化处理
        将X每一列都变成标准正态分布的形式，即满足均值为0，标准差为1
        标准化也叫标准差标准化，经过处理的数据符合标准正态分布。
        """
        return (X-self.mean_)/self.std_

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def restore(self,X):
        return X * self.std_ + self.mean_


# 数据标准化后的梯度下降
ss1 = StandardScaler()
X_train_std = ss1.fit_transform(X_train) # 训练并转换数据
X_test_std = ss1.transform(X_test) # 用X_train的均值和标准差转换X_test数据

ss2 = StandardScaler()
y_train_std = ss2.fit_transform(y_train)
y_test_std = ss2.transform(y_test)

# print("标准化前:",X_train[:5])
# print("标准化后:",X_train_std[:5])

lr = LinearRegression_GradientDescent(alpha=0.0001,times=20)
lr.fit(X_train_std,y_train_std)
y_predict_std = lr.predict(X_test_std)
print("y_predict_std shape:",y_predict_std.shape)
print("MSE:", np.mean(((y_predict_std-y_test_std)**2)) ) # 求均方误差MSE
print("求解得到的w:",lr.w_)
print("loss变化情况:",lr.loss)

"""
学习率alpha非常重要，在标准化之后，我将alpha设置为0.05和0.005，得到的MSE也非常非常大。
"""

# y值还原为标准化前
y_test_restore = ss2.restore(y_test_std)
y_predict_restore = ss2.restore(y_predict_std)

# 画出预测值与真实值，直观展示
mpl.rcParams["font.family"] = "SimHei" #显示中文
mpl.rcParams["axes.unicode_minus"] = False # 显示负号

# print(y_predict[:5],y_train[:5],y_test[:5])
plt.figure(figsize=(5,5))
plt.plot(y_predict_std,"ro-") # 红色圆圈实线，画出标准化后预测值
plt.plot(pd.DataFrame(y_test_std).values,"go-") # 绿色圆圈实线，画出标准化后真实值
plt.plot(y_predict_restore,"bo-") # 蓝色圆圈实线，画出标准化前预测值
plt.plot(pd.DataFrame(y_test_restore).values,"yo-") # 黄色圆圈实线，画出标准化前真实值
plt.title("LSM预测")
plt.xlabel("样本序号")
plt.ylabel("房价")
plt.legend()
plt.show()












