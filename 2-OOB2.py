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
                             '是否使用默认配置文件':'默认配置文件状况是否异常','account_type':'账号属性'})

import copy
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

y_train = train_c['账号属性']
X_train = train_c.iloc[:,:-1]

rf = RandomForestClassifier(oob_score=True,random_state=42)
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

# 信息熵
y_train = train_c['账号属性']
X_train = train_c.iloc[:,:-1]

rf = RandomForestClassifier(criterion='entropy',random_state=42)
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
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.datasets import make_classification

# Define the labels for the different classes
labels = [0, 1]

# Create a figure with subplots for each class
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

# Loop over each class and plot the OOB error rate for different numbers of trees
for i in range(2):
    # Create a list to store the OOB error rates for each number of trees
    oob_error_rates = []
    for n in range(1,100,10):
        # Create a random forest classifier with n trees
        clf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
        # Fit the classifier to the training data
        clf.fit(X_train, y_train)
        # Calculate the OOB error rate
        oob_error_rate = 1 - clf.oob_score_
        # Append the OOB error rate to the list
        oob_error_rates.append(oob_error_rate)
    # Plot the OOB error rates for the current class
    axs[i].plot(range(1,100,10), oob_error_rates)
    axs[i].set_xlabel('Number of Trees')
    axs[i].set_ylabel('OOB Error Rate')
    axs[i].set_title(labels[i])

# Show the plot
plt.show()



















































































