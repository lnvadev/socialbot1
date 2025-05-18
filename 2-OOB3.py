import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('D:/somefilelist/tuite_train.csv')
Target = pd.read_csv('D:/somefilelist/tuite_Target.csv')

train_c = pd.concat([train,Target],axis=1)

train_c = train_c.rename(columns={'default_profile':'是否具有默认图像配置文件','互关好友数量':'关注数',
                             'geo_enabled':'账号是否开通地理定位','verified':'账号是否被平台认证',
                             'account_age_days':'账号累计登陆天数','粉丝数量与互关好友数量比值':'关注数与被关注数的比值',
                             '互关好友数量与收藏点赞数比值':'收藏点赞数与关注数的比值','发文总量与互关好友数量比值':'发文总量与关注数的比值',
                             '账号个性签名是否异常':'账号描述是否异常','是否大量存在创建时间相同的账号':'账号是否为同一天创建',
                             '发文总数与收藏点赞数比值':'收藏点赞数与发文总量比值','累计登录天数与粉丝数量的比值':'粉丝数量与累计登录天数的比值',
                             '累计登录天数与收藏点赞数的比值':'收藏点赞数与累计登录天数的比值','累计登录天数与互关好友数量的比值 ':'关注数与累计登录天数的比值',
                             '默认配置文件状况是否异常':'是否使用默认配置文件','account_type':'账号属性'})

import copy
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

y_train = train_c['账号属性']
X_train = train_c.iloc[:,:-1]

x_train,x_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.1,random_state=29) 
rf = RandomForestClassifier(oob_score=True,random_state=29)
rf.fit(x_train, y_train)
print(rf.score(x_test,y_test))

y_train = train_c['账号属性']
X_train = train_c.iloc[:,:-1]
a = X_train['收藏点赞数'].sample(frac=1.0)
X = pd.DataFrame()
X['收藏点赞数'] = pd.Series(a)

X = X.reset_index(drop=True)

X_train.drop(['收藏点赞数'],axis=1,inplace=True)
X_train = pd.concat([X_train,X],axis=1)

x_train,x_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.1,random_state=29) 
rf = RandomForestClassifier(oob_score=True,random_state=29)
rf.fit(x_train, y_train)
# x_test,y_test
print(rf.score(x_test,y_test))
















