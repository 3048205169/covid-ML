import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score

def method1():
    UKcase = "new_dataframe-China.csv"
    upd = pd.read_csv(UKcase)
    upd = upd.loc[0:1001]
    upd.fillna(0,inplace=True)

    data = upd.drop('Unnamed: 0',axis=1)
    data = data.drop('iso_code',axis=1)
    data = data.drop('continent',axis=1)
    data = data.drop('location',axis=1)
    data = data.drop('date',axis=1)
    data = data.drop('total_cases',axis=1)
    data = data.drop('new_cases_smoothed',axis=1)
    data = data.drop('stringency_index',axis=1)

    target = upd[['new_cases']]

    # print(data.columns)

    x_train,x_test,y_train,y_test = train_test_split(data,target,random_state=30)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)


    estimator = LinearRegression()

    lr = estimator.fit(x_train,y_train)

    print("权重系数为: \n", estimator.coef_)
    print("偏置为: \n",estimator.intercept_)


    y_predict = lr.predict(x_test)
    # print("预测结果为: \n", y_predict)
    error = mean_squared_error(y_test,y_predict)

    print("正规方程-均方误差为:\n ",error)

    r2 = r2_score(y_test,y_predict)
    print("r2 = ",r2)

    t = np.arange(len(x_test))

    plt.plot(t,y_test,'r',linewidth=1,label='y_test')
    plt.plot(t,y_predict,'g',linewidth=1,label='y_train')
    plt.legend()
    plt.title("Chinese LinearRegression")
    plt.savefig("Chinese LinearRegression.jpg")
    plt.show()

def method2():
    UKcase = "new_dataframe-China.csv"
    upd = pd.read_csv(UKcase)
    upd = upd.loc[0:1001]
    upd.fillna(0,inplace=True)

    data = upd.drop('Unnamed: 0',axis=1)
    data = data.drop('iso_code',axis=1)
    data = data.drop('continent',axis=1)
    data = data.drop('location',axis=1)
    data = data.drop('date',axis=1)
    data = data.drop('total_cases',axis=1)
    data = data.drop('new_cases_smoothed',axis=1)
    data = data.drop('stringency_index',axis=1)

    target = upd[['new_cases']]

    # print(data.columns)

    x_train,x_test,y_train,y_test = train_test_split(data,target,random_state=30)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)


    estimator = Ridge()

    lr = estimator.fit(x_train,y_train)

    print("权重系数为: \n", estimator.coef_)
    print("偏置为: \n",estimator.intercept_)


    y_predict = lr.predict(x_test)
    # print("预测结果为: \n", y_predict)
    error = mean_squared_error(y_test,y_predict)

    print("正规方程-均方误差为:\n ",error)

    r2 = r2_score(y_test,y_predict)
    print("r2 = ",r2)

    t = np.arange(len(x_test))

    plt.plot(t,y_test,'r',linewidth=1,label='y_test')
    plt.plot(t,y_predict,'g',linewidth=1,label='y_train')
    plt.legend()
    plt.title("Chinese RidgeRegressor")
    plt.savefig("Chinese RidgeRegressor.jpg")
    plt.show()

if __name__ =="__main__":
    method1()
    print("#########################################")
    method2()