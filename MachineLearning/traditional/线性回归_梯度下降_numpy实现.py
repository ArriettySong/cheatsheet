import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

class LinearRegression_GradientDescent:

    def __init__(self,alpha,times):
        """:var
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
        # 权重w0调整公式：alpha * sum(y-y_hat)
        # 权重wi调整公式：alpha * sum((y-y_hat) * xi)
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


# 不考虑截距
boston = datasets.load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names
print("feature_names:",feature_names)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print("train data's shape:",X_train.shape)
print("test data's shape:",X_test.shape)

lr = LinearRegression_GradientDescent(alpha=0.05,times=10)
lr.fit(X_train,y_train)
y_predict = lr.predict(X_test)
print("y_predict shape:",y_predict.shape)
print("MSE:", np.mean(((y_predict-y_test)**2)) ) # 求均方误差MSE
print("求解得到的w:",lr.w_)
print("loss变化情况:",lr.loss)


# # 画出预测值与真实值，直观展示
# mpl.rcParams["font.family"] = "SimHei" #显示中文
# mpl.rcParams["axes.unicode_minus"] = False # 显示负号
#
# # print(y_predict[:5],y_train[:5],y_test[:5])
# plt.figure(figsize=(5,5))
# plt.plot(y_predict,"ro-") # 红色圆圈实线，画出预测值
# plt.plot(pd.DataFrame(y_test).values,"go-") # 绿色圆圈实线，画出真实值
# plt.title("LSM预测")
# plt.xlabel("样本序号")
# plt.ylabel("房价")
# plt.legend()
# plt.show()












