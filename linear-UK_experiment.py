import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score

def method1():
    UKcase = "new_dataframe-UK.csv"
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
    print("RMSE = ", np.sqrt(error))

    r2 = r2_score(y_test,y_predict)
    print("r2 = ",r2)

    t = np.arange(len(x_test))

    '''
        下面就是新的制图方式，将原本的true的data和用于训练以及预测的data一并绘入
        true data = target
        预测以及用于训练的数据
        y_test有每个index对应两个值，一个是代表的天数的，一个是代表new_cases
        而y_train和y_test结构相同

        而y_predict和y_test实际上一一对应，但是遗憾的是,y_predict的代表天数的
        数据是错误的，需要调整的和y_test一致才可以
        '''

    date = y_test.index

    y_predict = pd.DataFrame(y_predict)

    y_predict.index = date

    '''
    把y_predict和y_train融合起来
    '''
    y_predict.rename(columns={0: 'new_cases'}, inplace=True)

    y_prediction = y_predict.append(y_train)

    y_prediction = adjust_y_pred(y_prediction)

    # plt.plot(t,y_test,'r',linewidth=1,label='y_test')
    # plt.plot(t,y_predict,'g',linewidth=1,label='y_train')
    #
    #
    date = np.arange(len(target))
    plt.plot(date, y_prediction, 'g', linewidth=1, label='predict_value')
    plt.plot(date, target, 'r', linewidth=1, label='true_value')
    plt.xlabel("day")
    plt.ylabel("new cases each day")
    plt.legend()
    plt.title("UK LinearRegression")
    plt.savefig("UK LinearRegression.jpg")
    plt.show()

def method2():
    UKcase = "new_dataframe-UK.csv"
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

    print("RMSE = ", np.sqrt(error))

    r2 = r2_score(y_test,y_predict)
    print("r2 = ",r2)

    t = np.arange(len(x_test))

    '''
        下面就是新的制图方式，将原本的true的data和用于训练以及预测的data一并绘入
        true data = target
        预测以及用于训练的数据
        y_test有每个index对应两个值，一个是代表的天数的，一个是代表new_cases
        而y_train和y_test结构相同

        而y_predict和y_test实际上一一对应，但是遗憾的是,y_predict的代表天数的
        数据是错误的，需要调整的和y_test一致才可以
        '''

    date = y_test.index

    y_predict = pd.DataFrame(y_predict)

    y_predict.index = date

    '''
    把y_predict和y_train融合起来
    '''
    y_predict.rename(columns={0: 'new_cases'}, inplace=True)

    y_prediction = y_predict.append(y_train)

    y_prediction = adjust_y_pred(y_prediction)

    # plt.plot(t,y_test,'r',linewidth=1,label='y_test')
    # plt.plot(t,y_predict,'g',linewidth=1,label='y_train')
    #
    #
    date = np.arange(len(target))
    plt.plot(date, y_prediction, 'g', linewidth=1, label='predict_value')
    plt.plot(date, target, 'r', linewidth=1, label='true_value')
    plt.xlabel("day")
    plt.ylabel("new cases each day")
    plt.legend()
    plt.title("UK RidgeRegression")
    plt.savefig("UK RidgeRegression.jpg")
    plt.show()

def adjust_y_pred(y_prediction):
    '''
    将y_prediction的顺序按照日期填入
    '''
    res = y_prediction.sort_index(axis=0,inplace=False)
    '''
    把res第一列的数据全部合到第二列中去
    '''


    return res

if __name__ =="__main__":
    method1()
    print("#########################################")
    method2()