import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

class LinearRegression_LeastSquareMethod:

    def fit(self,X,y,feature_names,Intercept = False):
        # 如果求解的w需要截距，则对训练数据进行转换。
        if(Intercept):
            self.Intercept = True
            self.feature_names = feature_names
            X = data_add_column(X,self.feature_names)
        else:
            self.Intercept = False

        print("X原始数据的shape：",X.shape)
        X = np.asmatrix(X)  # 进行运算时需要将nparray转为matrix矩阵,不过，这里的x在转为矩阵前后的shape都是多少
        print("X转换为矩阵后的shape：",X.shape)
        y = np.asmatrix(y)
        print("y转换为矩阵后的shape：",y.shape)
        y = y.reshape(-1,1)
        print("y reshape后的shape：",y.shape)
        self.w_ = (X.T * X).I * X.T * y # 最小二乘法中，w的求解方法
        print("w的shape：",self.w_.shape)

        # demension为特征数，n_samples为样本数
        # 对于单个样本x，y = w.T * x   (x为列向量 demension*1)
        # 对于多个样本X，y = w.T * X.T = X * w （多个样本列向量 demension*n_samples）


    def predict(self,X):
        X = np.asmatrix(X.copy())
        if(self.Intercept):
            X = data_add_column(X,self.feature_names)
        y = X * self.w_     # 或者是w
        return np.array(y).ravel() # 返回的不可以是矩阵，需要是nparray



def data_add_column(X,feature_names):
    df = pd.DataFrame(X,columns=feature_names)
    new_columns = df.columns.insert(0,"Intercept")
    print("new_columns:",new_columns)
    # print(df.shape)
    df = df.reindex(columns = new_columns,fill_value=1)
    # print(df.shape)
    return np.asarray(df)

# 不考虑截距
boston = datasets.load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names
print("feature_names:",feature_names)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
print("train data's shape:",X_train.shape)
print("test data's shape:",X_test.shape)

lr = LinearRegression_LeastSquareMethod()
lr.fit(X_train,y_train,feature_names,False)
y_predict = lr.predict(X_test)
print("y_predict shape:",y_predict.shape)
print("MSE:", np.mean(((y_predict-y_test)**2)) ) # 求均方误差MSE
print("求解得到的w:",lr.w_)

# 考虑截距，需要为样本增加一维数据，其值均为1，对应参数为w0
lr = LinearRegression_LeastSquareMethod()
lr.fit(X_train,y_train,feature_names,True)
y_predict = lr.predict(X_test)
print("y_predict shape:",y_predict.shape)
print("MSE:", np.mean(((y_predict-y_test)**2)) ) # 求均方误差MSE
print("求解得到的w:",lr.w_)


# 画出预测值与真实值，直观展示
mpl.rcParams["font.family"] = "SimHei" #显示中文
mpl.rcParams["axes.unicode_minus"] = False # 显示负号

# print(y_predict[:5],y_train[:5],y_test[:5])
plt.figure(figsize=(5,5))
plt.plot(y_predict,"ro-") # 红色圆圈实线，画出预测值
plt.plot(pd.DataFrame(y_test).values,"go-") # 绿色圆圈实线，画出真实值
plt.title("LSM预测")
plt.xlabel("样本序号")
plt.ylabel("房价")
plt.legend()
plt.show()



"""
普通最小二乘的系数估计依赖于特征的独立性。
当特征相关且设计矩阵的列之间具有近似线性相关性时,设计矩阵趋于奇异矩阵，最小二乘估计对观测目标的随机误差高度敏感，可能产生很大的方差。
例如，在没有实验设计的情况下收集数据时，就可能会出现这种多重共线性的情况。
https://scikit-learn.org.cn/view/4.html#1.1.1%20%E6%99%AE%E9%80%9A%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95
"""









