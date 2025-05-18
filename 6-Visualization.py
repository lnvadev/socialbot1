import numpy as np
import pandas as pd
train = pd.read_csv('tuite_train.csv')
train['created_at'] = pd.to_datetime(train['created_at'])

def human(month):
    if month == 1: return '人类账号';  #人类
    if month == 0: return '机器人账号'; 
    return 2;

train['account_type'] = train.apply(lambda x: human(x.account_type),axis=1)
train = train.rename(columns={'account_type':'账号属性'})

list1 = [i for i in range(1,35493)]
train['序号'] = pd.DataFrame(list1)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
train1 = train.loc[train['账号属性']=='人类账号']
plt.scatter(x='account_age_days', y='收藏点赞数',c="blue",data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("累计登录天数")
plt.ylabel("点赞数")
plt.ylim(0,1000000)

train1 = train.loc[train['账号属性']=='机器人账号']
plt.scatter(x='account_age_days', y='收藏点赞数',c=plt.cm.Set1(4),data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("累计登录天数")
plt.ylabel("点赞数")
plt.ylim(0,1000000)

palette={'机器人账号': 'orange', '人类账号': 'b'}
ax = sns.catplot(train['账号属性'],train['平均每天发文数量'],palette=palette, jitter=False,data=train)

ax = sns.catplot(train['账号属性'],train['发文总量'],jitter=False,palette=palette,data=train)
train.info()

train1 = train.loc[train['账号属性']=='人类账号']
plt.scatter(x='收藏点赞数', y='发文总量',c="blue",data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("收藏点赞数")
plt.ylabel("发文总量")
plt.xlim(0,800000)
plt.ylim(0,2000000)

train1 = train.loc[train['账号属性']=='机器人账号']
plt.scatter(x='收藏点赞数', y='发文总量',c=plt.cm.Set1(4),data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("收藏点赞数")
plt.ylabel("发文总量")
plt.xlim(0,800000)
plt.ylim(0,2000000)

train = train.rename(columns={'发文总量与互关好友数量比值':'发文总量与关注数的比值'})

cols = ['发文总量与关注数的比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="发文总量与关注数的比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']

list1 = [i for i in range(0,21710)]
list_c = list(test1["发文总量与关注数的比值"])
list_c.sort(key=None, reverse=False) 

list2 = [i for i in range(0,8285)]
list_c2 = list(test2["发文总量与关注数的比值"])
list_c2.sort(key=None, reverse=False) 

#sns.set(color_codes=True)#导入seaborn包设定颜色
fig,ax=plt.subplots(figsize=(10,5),dpi=520)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("发文总量与关注数的比值",fontproperties='SimHei',fontsize=10)
sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)
sns.set(style="ticks")
plt.rcParams['font.sans-serif']=['SimHei']
plt.legend()

cols = ['发文总数与粉丝数量比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="发文总数与粉丝数量比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']

list1 = [i for i in range(0,22501)]
list_c = list(test1["发文总数与粉丝数量比值"])
list_c.sort(key=None, reverse=False) 


list2 = [i for i in range(0,8490)]
list_c2 = list(test2["发文总数与粉丝数量比值"])
list_c2.sort(key=None, reverse=False) 


fig,ax=plt.subplots(figsize=(10,5),dpi=520)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("发文总数与粉丝数量比值",fontproperties='SimHei',fontsize=10)
sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)
sns.set(style="ticks")
plt.rcParams['font.sans-serif']=['SimHei']
plt.legend()

train.info()

cols = ['互关好友数量与收藏点赞数比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="互关好友数量与收藏点赞数比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']

len(test2)
list1 = [i for i in range(0,19852)]
list_c = list(test1["互关好友数量与收藏点赞数比值"])
list_c.sort(key=None, reverse=False) 


list2 = [i for i in range(0,10041)]
list_c2 = list(test2["互关好友数量与收藏点赞数比值"])
list_c2.sort(key=None, reverse=False) 

fig,ax=plt.subplots(figsize=(10,5),dpi=520)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("点赞数与关注数比值",fontproperties='SimHei',fontsize=10)

sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)
sns.set(style="ticks")
plt.rcParams['font.sans-serif']=['SimHei']
plt.legend()

cols = ['收藏点赞数与发文总数比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="收藏点赞数与发文总数比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']

list1 = [i for i in range(0,20803)]
list_c = list(test1["收藏点赞数与发文总数比值"])
list_c.sort(key=None, reverse=False) 


list2 = [i for i in range(0,10180)]
list_c2 = list(test2["收藏点赞数与发文总数比值"])
list_c2.sort(key=None, reverse=False) 

fig,ax=plt.subplots(figsize=(10,5),dpi=520)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("点赞数与发文总数比值",fontproperties='SimHei',fontsize=10)

sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)
sns.set(style="ticks")
plt.rcParams['font.sans-serif']=['SimHei']
plt.legend()
