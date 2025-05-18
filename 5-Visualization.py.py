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

#pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 100  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题
plt.rcParams['font.sans-serif']=['SimHei']

train1 = train.loc[train['账号属性']=='人类账号']
plt.scatter(x='account_age_days', y='粉丝数量',c="blue",data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("累计登录天数")
plt.ylabel("粉丝数量")
plt.ylim(0,150000000)

train1 = train.loc[train['账号属性']=='机器人账号']
plt.scatter(x='account_age_days', y='粉丝数量',c=plt.cm.Set1(4),data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("累计登录天数")
plt.ylabel("粉丝数量")
plt.ylim(0,150000000)

train1 = train.loc[train['账号属性']=='人类账号']
plt.scatter(x='account_age_days', y='收藏点赞数',c="blue",data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("累计登录天数")
plt.ylabel("点赞数")
plt.ylim(0,3000000)

train1 = train.loc[train['账号属性']=='机器人账号']
plt.scatter(x='account_age_days', y='收藏点赞数',c=plt.cm.Set1(4),data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("累计登录天数")
plt.ylabel("点赞数")
plt.ylim(0,3000000)

train.info()

train1 = train.loc[train['账号属性']=='人类账号']
plt.scatter(x='account_age_days', y='关注数量与被关注数的比值',c="blue",data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("累计登录天数")
plt.ylabel("关注数量与被关注数的比值")
plt.ylim(0,300)

train1 = train.loc[train['账号属性']=='机器人账号']
plt.scatter(x='account_age_days', y='关注数量与被关注数的比值',c=plt.cm.Set1(4),data=train1, edgecolors=None, alpha = 0.7)
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("累计登录天数")
plt.ylabel("关注数量与被关注数的比值")
plt.ylim(0,300)

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

sns.set(color_codes=True)#导入seaborn包设定颜色
fig,ax=plt.subplots(figsize=(10,5),dpi=120)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("发文总量与关注数的比值",fontproperties='SimHei',fontsize=10)
plt.rcParams['font.sans-serif']=['SimHei']

sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)

plt.legend()

# 互关好友数量与收藏点赞数比值
train = train.rename(columns={'互关好友数量与收藏点赞数比值':'收藏点赞数与关注数的比值'})
cols = ['收藏点赞数与关注数的比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="收藏点赞数与关注数的比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']

list1 = [i for i in range(0,19852)]
list_c = list(test1["收藏点赞数与关注数的比值"])
list_c.sort(key=None, reverse=False) 


list2 = [i for i in range(0,10041)]
list_c2 = list(test2["收藏点赞数与关注数的比值"])
list_c2.sort(key=None, reverse=False) 

sns.set(color_codes=True)#导入seaborn包设定颜色
fig,ax=plt.subplots(figsize=(10,5),dpi=120)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("收藏点赞数与关注数的比值",fontproperties='SimHei',fontsize=10)
plt.rcParams['font.sans-serif']=['SimHei']

sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)

plt.legend()

cols = ['发文总数与粉丝数量比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="发文总数与粉丝数量比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']

len(test2)
list1 = [i for i in range(0,22501)]
list_c = list(test1["发文总数与粉丝数量比值"])
list_c.sort(key=None, reverse=False) 


list2 = [i for i in range(0,8490)]
list_c2 = list(test2["发文总数与粉丝数量比值"])
list_c2.sort(key=None, reverse=False) 

sns.set(color_codes=True)#导入seaborn包设定颜色
fig,ax=plt.subplots(figsize=(10,5),dpi=120)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("发文总数与粉丝数量比值",fontproperties='SimHei',fontsize=10)
plt.rcParams['font.sans-serif']=['SimHei']

sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)

plt.legend()

cols = ['收藏点赞数与发文总数比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="收藏点赞数与发文总数比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']
len(test2)
list1 = [i for i in range(0,20803)]
list_c = list(test1["收藏点赞数与发文总数比值"])
list_c.sort(key=None, reverse=False) 


list2 = [i for i in range(0,10180)]
list_c2 = list(test2["收藏点赞数与发文总数比值"])
list_c2.sort(key=None, reverse=False) 

sns.set(color_codes=True)#导入seaborn包设定颜色
fig,ax=plt.subplots(figsize=(10,5),dpi=120)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("收藏点赞数与发文总数比值",fontproperties='SimHei',fontsize=10)
plt.rcParams['font.sans-serif']=['SimHei']

sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)

plt.legend()
train.info()

cols = ['粉丝数量与累计登录天数的比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="粉丝数量与累计登录天数的比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']


len(test2)
list1 = [i for i in range(0,16780)]
list_c = list(test1["粉丝数量与累计登录天数的比值"])
list_c.sort(key=None, reverse=False) 


list2 = [i for i in range(0,10988)]
list_c2 = list(test2["粉丝数量与累计登录天数的比值"])
list_c2.sort(key=None, reverse=False) 

sns.set(color_codes=True)#导入seaborn包设定颜色
fig,ax=plt.subplots(figsize=(10,5),dpi=120)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("粉丝数量与累计登录天数的比值",fontproperties='SimHei',fontsize=10)
plt.rcParams['font.sans-serif']=['SimHei']

sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)

plt.legend()

# 互关好友数量与累计登录天数的比值
train = train.rename(columns={'互关好友数量与累计登录天数的比值':'关注数与累计登录天数的比值'})
cols = ['关注数与累计登录天数的比值'] # one or more
Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1
train_c = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train_c.sort_values(by="关注数与累计登录天数的比值" , inplace=True, ascending=True) 

test1 = train_c.loc[train_c['账号属性'] == '人类账号']
test2 = train_c.loc[train_c['账号属性'] == '机器人账号']

len(test2)
list1 = [i for i in range(0,20507)]
list_c = list(test1["关注数与累计登录天数的比值"])
list_c.sort(key=None, reverse=False) 


list2 = [i for i in range(0,11056)]
list_c2 = list(test2["关注数与累计登录天数的比值"])
list_c2.sort(key=None, reverse=False) 

sns.set(color_codes=True)#导入seaborn包设定颜色
fig,ax=plt.subplots(figsize=(10,5),dpi=120)
plt.xlabel("序号",fontproperties='SimHei',fontsize=10)
plt.ylabel("关注数与累计登录天数的比值",fontproperties='SimHei',fontsize=10)
plt.rcParams['font.sans-serif']=['SimHei']

sns.lineplot(list1,list_c,label='人类账号', markers=True)
sns.lineplot(list2,list_c2,label='机器人账号', markers=True)

plt.legend()













































