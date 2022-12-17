import datetime

import pandas as pd
pd.set_option('expand_frame_repr',False)

ukPolicyData = "coronanet_release_United Kingdom.csv"

upd = pd.read_csv(ukPolicyData)

startTime = '2020/1/30'
startTime = datetime.datetime.strptime(startTime,"%Y/%m/%d")

print(startTime)



new_data = upd[['date_start','date_end','type']]

u_type = upd[['type']]
# print(new_data)


# print(new_data)

cases_raw = pd.read_csv("United_Kingdom.csv")

# processed = cases_raw[['date','newCasesBySpecimenDate','cumCasesBySpecimenDate']]
#
# cases = processed.iloc[::-1]

cases = cases_raw
#先在cases中添加所有的政策的列

col = []

for t in u_type['type'].unique():
    col.append(t)


for i in range(0,len(col)):
    cases.insert(loc = len(cases.columns),column=col[i],value=0)


new_data['date_start'] = new_data['date_start'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))


new_data['date_end'] =pd.to_datetime(new_data['date_end'],format='%Y-%m-%d')
# new_data['date_end'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))


new_data = new_data.dropna()


cases['date'] = cases['date'].apply(lambda x:datetime.datetime.strptime(x,'%Y/%m/%d'))

#############################

'''
这一步把cases加入一行代表上一天的感染人数，这是方便后期做得好
'''

daily = cases['new_cases'].tolist()
del daily[0]
prev_daily = pd.DataFrame(daily)
cases.insert(loc=0,column="prev_daily",value=prev_daily)

print("here")

def getDate(cases, date):
    one_day = datetime.timedelta(days=1)
    timeDelta = date-startTime
    return 986-timeDelta.days
    # for i in cases.iterrows():
    #     curDate = i[1]['date']
    #     if curDate == date:
    #         return i[0]

def setPolicy(rowStart, rowEnd, policy,cases):
    date = rowStart
    one_day = datetime.timedelta(days=1)
    while(date<=rowEnd):
        date = date+one_day
        row = getDate(cases,date)
        cases.loc[row,policy] = 1




for policy in new_data.iterrows():
    start = policy[1]['date_start']
    end = policy[1]['date_end']
    if(start == 'NaN'):
        continue
    if(end == 'NaN'):
        continue
    policy = policy[1]['type']
    setPolicy(start,end,policy,cases)


########
####cases还要加上前一天的感染人数和累计感染人数

cases.dropna()



cases.to_csv("new_dataframe-UK.csv")



'''
参数：
1。每日新增感染人数
2。当前政策
3。是否强制
'''


