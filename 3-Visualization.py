import numpy as np
import pandas as pd

train = pd.read_csv('tuite_train.csv')
train['created_at'] = pd.to_datetime(train['created_at'])
list1 = [i for i in range(1,35493)]
train['序号'] = pd.DataFrame(list1)
# train = train.rename(columns={'':"收藏文件数量与发文总数的比值"})
list2 = list(train['粉丝数量与互关好友数量比值'])
list2.sort(reverse=False)
train['粉丝与好友'] = pd.DataFrame(list2)

def human(month):
    if month == 1: return '人类账号';  #人类
    if month == 0: return '机器人账号'; 
    return 2;

train['account_type'] = train.apply(lambda x: human(x.account_type),axis=1)
train = train.rename(columns={'account_type':'账号属性'})

# 发文总数与粉丝数量比值
sns.relplot(x='序号' ,y='粉丝与好友',hue='账号属性',kind="line",markers=True, data=train)

# 关注数与被关注数

train.info()
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def human(month):
    if month == 1: return '人类账号';  #人类
    if month == 0: return '机器人账号'; 
    return 2;

train['account_type'] = train.apply(lambda x: human(x.account_type),axis=1)
train = train.rename(columns={'account_type':'账号属性'})

train1 = train.sample(n=8000)

# 设置
pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 100  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题

# 同时设置hue,size
plt.rcParams['font.sans-serif']=['SimHei']
sns.scatterplot(data=train1,x='序号' ,y='粉丝数量与互关好友数量比值',hue='账号属性')
plt.ylim(0,100000)
# plt.legend(["一区","二区","三区","四区"], loc='upper center',bbox_to_anchor=(1.1,1.1))
plt.show()

train.info()

# 累计登录天数与粉丝数量的比值

pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 100  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题

# 同时设置hue,size
plt.rcParams['font.sans-serif']=['SimHei']
sns.scatterplot(data=train1,x='序号' ,y='累计登录天数与粉丝数量的比值',hue='账号属性')
# plt.legend(["一区","二区","三区","四区"], loc='upper center',bbox_to_anchor=(1.1,1.1))
#plt.ylim(0,1800)
plt.show()

# 发文总数与粉丝数量比值
pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 100  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题

# 同时设置hue,size
plt.rcParams['font.sans-serif']=['SimHei']
sns.scatterplot(data=train1,x='序号' ,y='发文总数与粉丝数量比值',hue='账号属性')
# plt.legend(["一区","二区","三区","四区"], loc='upper center',bbox_to_anchor=(1.1,1.1))

plt.ylim(0,5000)
plt.show()

# 发文总数与粉丝数量比值
sns.relplot(x='序号' ,y='发文总数与粉丝数量比值',hue='账号属性',kind="line",markers=True, data=train)

sns.relplot(x='序号' ,y='发文总数与收藏点赞数比值',hue='账号属性',kind="line",markers=True, data=train)

# 
pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 100  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题

# 同时设置hue,size
plt.rcParams['font.sans-serif']=['SimHei']
sns.scatterplot(data=train,x='序号' ,y='发文总数与收藏点赞数比值',hue='账号属性')
plt.ylim(0,100000)
# plt.legend(["一区","二区","三区","四区"], loc='upper center',bbox_to_anchor=(1.1,1.1))
plt.show()

# 收藏点赞数

pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 100  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题

# 同时设置hue,size
plt.rcParams['font.sans-serif']=['SimHei']
sns.scatterplot(data=train,x='序号' ,y='平均每天发文数量',hue='账号属性')
# plt.legend(["一区","二区","三区","四区"], loc='upper center',bbox_to_anchor=(1.1,1.1))
plt.show()

sns.relplot(x='序号' ,y='平均每天发文数量',hue='账号属性',kind="line",markers=True, data=train)

sns.relplot(x='序号' ,y='平均每天发文数量',hue='账号属性',kind="line",markers=True,data=train)

g = sns.relplot(x='created_at' ,y='平均每天发文数量',hue='账号属性', kind="line", data=train)
g.fig.autofmt_xdate() #调节横轴标签方向











