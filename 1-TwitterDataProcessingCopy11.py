import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('D:/somefilelist/tuite_train.csv')
Target = pd.read_csv('D:/somefilelist/tuite_Target.csv')

train_c = pd.concat([train,Target],axis=1)

train_c = train_c.rename(columns={'default_profile':'默认图像配置文件是否异常','互关好友数量':'关注数','收藏点赞数':'点赞数',
                                 '是否使用默认配置文件':'默认配置文件是否异常',
                             'geo_enabled':'账号是否开通地理定位','verified':'账号是否被平台认证',
                             'account_age_days':'账号累计登陆天数','粉丝数量与互关好友数量比值':'关注数与被关注数的比值',                       
                             '互关好友数量与收藏点赞数比值':'点赞数与关注数的比值','发文总量与互关好友数量比值':'发文总量与关注数的比值',   
                             '账号个性签名是否异常':'账号描述是否异常','是否大量存在创建时间相同的账号':'账号是否为同一天创建',
                             '发文总数与收藏点赞数比值':'点赞数与发文总量比值','累计登录天数与粉丝数量的比值':'粉丝数量与累计登录天数的比值',
                             '累计登录天数与收藏点赞数的比值':'点赞数与累计登录天数的比值','累计登录天数与互关好友数量的比值':'关注数与累计登录天数的比值',
                             '是否使用默认配置文件':'默认配置文件状况是否异常'})

import copy
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

y_train = train_c['account_type']
X_train = train_c.iloc[:,:-1]

rf = RandomForestClassifier(criterion='gini',oob_score=True,random_state=29)
rf.fit(X_train, y_train)

a = []
b = []
for feat, importance in zip(train_c.columns, rf.feature_importances_): 
    a.append(feat)
    b.append(importance)
import pandas as pd
aa2 = {'feature':a,'importance':b}
quanzhong = pd.DataFrame(aa2)
new_quanzhong = quanzhong.sort_values(by = 'importance',axis = 0,ascending = False)
new_quanzhong.to_csv(r"D:\somefilelist\shujuji\paixu.csv",index=False,encoding="utf_8_sig")
paixu= pd.read_csv('D:\somefilelist\shujuji\paixu.csv')

paixu.iloc[:,:]

x_list = []
for i in list(paixu['importance']):
    c = round(i,5)
    x_list.append(c)

y_list = ['111', '102', '110', '113', '103', '101',
 '112', '108', '119', '107', '106', '104',
 '105','109', '121', '115', '116',
 '120', '118', '117', '114']

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

OOB = pd.DataFrame()
OOB['编号'] = pd.Series(y_list)
OOB["相对重要性百分比/%"] = pd.Series(x_list)

ax = sns.barplot(x = '相对重要性百分比/%', y = '编号',data = OOB, palette="muted")
sns.set(style="ticks")
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.sans-serif']=['SimHei']
ax.set_yticklabels(labels = list(OOB['编号']),fontsize = 11) # 放大纵轴坐标
Confirmed1 = list(OOB["相对重要性百分比/%"])
for i in range(len(OOB['编号'])):    
    ax.text(Confirmed1[i],i,(lambda x:format(x,','))(Confirmed1[i]),color="black",ha="center", va='top',fontsize = 6)#添加数字标注，注意用lambda函数加千分位符
plt.show()


































































