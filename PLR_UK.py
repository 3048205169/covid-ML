import pandas as pd
import matplotlib.pyplot as plt


def pre_process(china_cases):
    china_cases = china_cases.drop('iso_code', axis=1)
    china_cases = china_cases.drop('continent', axis=1)
    china_cases = china_cases.drop('location', axis=1)
    china_cases = china_cases.drop('total_cases', axis=1)
    china_cases = china_cases.drop('new_cases_smoothed', axis=1)
    china_cases = china_cases.drop('total_deaths', axis=1)
    china_cases = china_cases.drop('new_deaths', axis=1)
    china_cases = china_cases.drop('reproduction_rate', axis=1)
    china_cases = china_cases.drop('stringency_index', axis=1)


    date = china_cases["date"]
    new_cases = china_cases["new_cases"]
    new_cases = new_cases.fillna(value=0)

    return date,new_cases

def make_date_cases_plot(date,new_cases):
    plt.plot(date, new_cases)
    plt.legend()
    plt.show()

def set_oxford_data():
    time = []
    for i in range(0,21):
        time.append(i)

    new_cases = [26,66,162,388,894,1952,
                 4143,8533,17268,33355,61833,
                 108538,179559,286946,448815,
                 679650,1014016,1468357,2169451,
                 3306749,5426740]
    return time,new_cases


def compress_china_cases(china_cases,rate):
    res = pd.DataFrame(columns=china_cases.columns)
    for i in range(0,len(china_cases)):
        if(i % rate == 0):
            res = res.append(china_cases.iloc[i])

    return res

def get_S(data):
    new_cases = data[1]
    if(type(new_cases)==pd.Series):
        new_cases = new_cases.tolist()
    s = []
    for i in range(0,len(new_cases)-1):
        s.append(new_cases[i+1]-new_cases[i])
    return s

def calculate_D(s1,s2):
    d = 0
    for i in range(0,len(s1)):
        d = d + abs(s1[i]-s2[i])

    return d


def do_PLR(real_data,Oxford_data):
    s1 = get_S(real_data)
    s2 = get_S(Oxford_data)
    D = calculate_D(s1,s2)
    return D

def set_my_data():
    new_cases = [0,3, 12, 48, 192, 768, 3072,
                 1228.8000000000002, 491.5200000000001,
                 196.60800000000006, 78.64320000000002,
                 31.45728000000001, 12.582912000000004,
                 5.033164800000002, 2.013265920000001,
                 0.8053063680000003, 0.32212254720000016,
                 0.12884901888000008, 0.05153960755200003,
                 0.020615843020800013, 0.008246337208320006,
                 0.003298534883328003, 0.0013194139533312013,
                 0.0005277655813324806, 0.00021110623253299223,
                 8.444249301319689e-05, 3.377699720527876e-05,
                 1.3510798882111505e-05, 5.404319552844602e-06,
                 2.161727821137841e-06, 8.646911284551364e-07]

    date = []
    for i in range(0,len(new_cases)):
        date.append(i)

    return (date,new_cases)

if __name__ =="__main__":
    china_cases = pd.read_csv("United_Kingdom.csv")
    china_cases = china_cases.loc[0:480]
    china_cases_1 = compress_china_cases(china_cases,24)

    date, new_cases = pre_process(china_cases_1)
    make_date_cases_plot(date, new_cases)
    date = []
    for i in range(0,len(date)):
        date.append(i)
    real_data = (date, new_cases)

    Oxford_data = set_oxford_data()
    # make_date_cases_plot(Oxford_data[0],Oxford_data[1])

    plr = do_PLR(real_data,Oxford_data)
    print("oxford plr = ",plr)


    my_data = set_my_data()

    china_cases_2 = compress_china_cases(china_cases,16)

    date, new_cases = pre_process(china_cases_2)

    real_data = (date,new_cases)

    plr = do_PLR(real_data,my_data)
    print("my_data plr = ",plr)



