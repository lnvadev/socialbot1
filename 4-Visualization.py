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

# 社交行为
pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 100  # 图形分辨率
sns.set_theme(style='darkgrid')  # 图形主题
plt.rcParams['font.sans-serif']=['SimHei']

sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 100
ax = sns.catplot(train['账号属性'],train['粉丝数量'],data=train)
plt.rcParams['font.sans-serif']=['SimHei']
# ax.set(ylim=(0, 1000000))

#train = train.rename(columns={'互关好友数量':'关注数量'})
sns.set(style="ticks")
plt.rcParams['font.sans-serif']=['SimHei']
ax = sns.catplot(train['账号属性'],train['关注数量'],data=train)

ax = sns.violinplot(train['账号属性'],train['互关好友数量'])
ax.set(ylim=(0, 4000))

# jitter=False
train = train.rename(columns={'粉丝数量与互关好友数量比值':'关注数与被关注数比值'})
ax = sns.catplot(train['账号属性'],train['关注数与被关注数比值'],jitter=False,data=train)
#ax.set(ylim=(0, 50000))

# 粉丝数量与互关好友数量比值
# ax = sns.violinplot(train['账号属性'],train['关注数与被关注数比值'],
#             data=train)
# ax.set(ylim=(0, 500))

ax = sns.catplot(train['账号属性'],train['平均每天发文数量'],jitter=False,data=train)

ax = sns.catplot(train['账号属性'],train['发文总量与互关好友数量比值'],jitter=False,data=train)

# 账号特征

cols = ['互关好友数量与累计登录天数的比值'] # one or more

Q1 = train[cols].quantile(0.25)
Q3 = train[cols].quantile(0.75)
IQR = Q3 - Q1

train = train[~((train[cols] < (Q1 - 1.5 * IQR)) |(train[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

train = train.rename(columns={'created_at':'账号创建时间'})
train.sort_values(by="发文总数与粉丝数量比值" , inplace=True, ascending=True) 
list1 = [i for i in range(1,35493)]
train['序号'] = pd.DataFrame(list1)
ax = sns.lineplot(x='账号创建时间',y='发文总数与粉丝数量比值',hue='账号属性', data=train)
# 
ax = sns.relplot(x='账号创建时间',y='收藏点赞数与发文总数比值',hue='账号属性',kind="line",markers=True, data=train)
#ax.set(ylim=(0, 1000))
sns.relplot(x='账号创建时间' ,y='粉丝数量与累计登录天数的比值',hue='账号属性', kind="line", data=train)
# 互关好友数量与累计登录天数的比值
sns.relplot(x='账号创建时间' ,y='收藏点赞数与累计登录天数的比值',hue='账号属性', kind="line", data=train)

# 雷达图
train.info()
train = train.rename(columns={'是否大量存在创建时间相同的账号':'是否为同一天创建'})
train = train.rename(columns={'账号个性签名是否异常':'账号描述是否异常'})
train = train.rename(columns={'默认配置文件状况是否异常':'是否使用默认配置文件'})
train = train.rename(columns={'verified':'账号是否被平台认证'})
train.groupby(['账号属性','是否为同一天创建']).size()
train.groupby(['账号属性','账号描述是否异常']).size()

train.groupby(['账号属性','地理定位是否异常']).size()
train.groupby(['账号属性','是否使用默认配置文件']).size()
train.groupby(['账号属性','账号是否被平台认证']).size()
# 社交机器人的情况

import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号"-"显示为方块的问题
#plt.style.use("ggplot")  # 设置ggplot样式

# 原始数据集并获取数据集长度

# 第一列是异常，第二列是正常
results = [{"是否为同一天创建": 4248,  "账号描述是否异常": 6282, "地理定位是否异常": 5276,
            "默认配置文件是否异常": 7053, '账号是否被平台认证': 11501},
           {"是否为同一天创建": 7550,  "账号描述是否异常": 5516, "地理定位是否异常": 6522,
            "默认配置文件是否异常": 4745, '账号是否被平台认证': 297}
           ]
data_length = len(results[0])
angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)  # 将极坐标根据数据长度进行等分

# 分离属性字段和数据
labels = [key for key in results[0].keys()]
score = [[v for v in result.values()] for result in results]

# 使雷达图数据封闭
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((labels, [labels[0]]))
score_Harry = np.concatenate((score[0], [score[0][0]]))
score_Son = np.concatenate((score[1], [score[1][0]]))

# 设置图形的大小
fig = plt.figure(figsize=(10, 6), dpi=500)

# 新建一个子图
ax = plt.subplot(111, polar=True)

# 绘制雷达图并填充颜色
ax.plot(angles, score_Harry, color="orange")
ax.fill(angles, score_Harry, "y", alpha=0.4)
ax.plot(angles, score_Son, color="b")
ax.fill(angles, score_Son, "cyan", alpha=0.4)

# 设置雷达图中每一项的标签显示
ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=15)
ax.set_theta_zero_location("E")  # 设置0度坐标轴起始位置，东西南北
# ax.set_rlim(0, 100)  # 设置雷达图的坐标刻度范围
#ax.set_rlabel_position(270)  # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
# ax.set_title("热刺球员能力对比图")
plt.legend(["状态异常数量","状态正常数量"], loc='upper center',bbox_to_anchor=(1.1,1.1))
plt.grid(True)
plt.rcParams['font.sans-serif']=['SimHei']
plt.show()

# 正常用户
import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams["font.sans-serif"] = ["KaiTi"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号"-"显示为方块的问题
plt.style.use("ggplot")  # 设置ggplot样式

# 原始数据集并获取数据集长度
results = [{"是否为同一天创建": 9766,  "账号描述是否异常": 5802, "地理定位是否异常": 3717,
            "默认配置文件是否异常": 7439, '账号是否被平台认证': 16746},
           {"是否为同一天创建": 13928,  "账号描述是否异常": 17892, "地理定位是否异常": 19977,
            "默认配置文件是否异常": 16255, '账号是否被平台认证': 6948}
           ]
data_length = len(results[0])

angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)  # 将极坐标根据数据长度进行等分

# 分离属性字段和数据
labels = [key for key in results[0].keys()]
score = [[v for v in result.values()] for result in results]

# 使雷达图数据封闭
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((labels, [labels[0]]))
score_Harry = np.concatenate((score[0], [score[0][0]]))
score_Son = np.concatenate((score[1], [score[1][0]]))

# 设置图形的大小
fig = plt.figure(figsize=(10, 6), dpi=500)

# 新建一个子图
ax = plt.subplot(111, polar=True)

# 绘制雷达图并填充颜色
ax.plot(angles, score_Harry, color="orange")
ax.fill(angles, score_Harry, "y", alpha=0.4)
ax.plot(angles, score_Son, color="b")
ax.fill(angles, score_Son, "cyan", alpha=0.4)


# 设置雷达图中每一项的标签显示
ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=15)

ax.set_theta_zero_location("E")  # 设置0度坐标轴起始位置，东西南北

# ax.set_rlim(0, 100)  # 设置雷达图的坐标刻度范围
ax.set_rlabel_position(270)  # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
# ax.set_title("热刺球员能力对比图")
plt.legend(["状态异常数量","状态正常数量"], loc='upper center',bbox_to_anchor=(1.1,1.1))
plt.show()




















