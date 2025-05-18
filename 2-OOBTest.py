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

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif']=['SimHei']

X = train_c.iloc[:,:10]
y = train_c['account_type']
print("特征名:", X.columns)

# 数据转为数据表
# X_df = pd.DataFrame(data=X, columns=X.columns)
X_df = X
X_df['账号属性'] = y

# 求相关性
data_coor = np.corrcoef(X_df.values, rowvar=0)
# columns=X_df.columns, index=X_df.columns
data_coor = pd.DataFrame(data=data_coor)
plt.figure(figsize=(8, 6), facecolor='w', dpi=1000) # 底色white
ax = sns.heatmap(data_coor, square=True, annot=True, fmt='.2f', 
                 linewidth=1, cmap='coolwarm_r',linecolor='white', cbar=True,
                 annot_kws={'size':10,'weight':'normal','color':'white'},
                 cbar_kws={'fraction':0.046, 'pad':0.03},
                 yticklabels=X_df.columns,  # 列标签
                xticklabels=X_df.columns ) 

plt.xticks(rotation=-90)  # x轴的标签旋转45度
plt.savefig("heatmap1.png", bbox_inches='tight')
plt.show()

# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif']=['SimHei']
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False

X = train_c.iloc[:,10:-1]
y = train_c['account_type']
print("特征名:", X.columns)

# 数据转为数据表
# X_df = pd.DataFrame(data=X, columns=X.columns)
X_df = X
X_df['账号属性'] = y

# 求相关性
data_coor = np.corrcoef(X_df.values, rowvar=0)
# columns=X_df.columns, index=X_df.columns
data_coor = pd.DataFrame(data=data_coor)
plt.figure(figsize=(8, 6), facecolor='w', dpi=1000) # 底色white
ax = sns.heatmap(data_coor, square=True, annot=True, fmt='.2f', 
                 linewidth=1, cmap='coolwarm_r',linecolor='white', cbar=True,
                 annot_kws={'size':10,'weight':'normal','color':'white'},
                 cbar_kws={'fraction':0.046, 'pad':0.03},
                 yticklabels=X_df.columns,  # 列标签
                xticklabels=X_df.columns ) 

plt.xticks(rotation=-90)  # x轴的标签旋转45度
plt.savefig("heatmap1.png", bbox_inches='tight')
plt.show()

train_x,test_x,train_y,test_y = train_test_split(train,Target,train_size=0.90,random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd

# Split data into features and target
X = train_x
y = train_y

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)

# Train the classifier
rfc.fit(X, y)

# Use out-of-bag (OOB) error estimate to select important features
sfm = SelectFromModel(rfc)
sfm.fit(X, y)

# Print selected features
selected_features = X.columns[sfm.get_support()]
print(selected_features)

import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('D:/somefilelist/tuite_train.csv')
Target = pd.read_csv('D:/somefilelist/tuite_Target.csv')

training,valid,y_train,y_valid = train_test_split(train,Target,train_size=0.90,random_state=42)

training = training[['收藏点赞数', '粉丝数量', '互关好友数量', 'account_age_days', '发文总数与收藏点赞数比值',
       '发文总数与粉丝数量比值', '累计登录天数与粉丝数量的比值', '累计登录天数与收藏点赞数的比值', '累计登录天数与互关好友数量的比值']]

valid = valid[['收藏点赞数', '粉丝数量', '互关好友数量', 'account_age_days', '发文总数与收藏点赞数比值',
       '发文总数与粉丝数量比值', '累计登录天数与粉丝数量的比值', '累计登录天数与收藏点赞数的比值', '累计登录天数与互关好友数量的比值']]

import numpy as np
clf = RandomForestClassifier()
clf.fit(training,y_train)
predicted = np.array(clf.predict_proba(valid))
rate = clf.score(valid, y_valid)
print("准确率为：%f" % rate)

from sklearn.metrics import classification_report,accuracy_score #评估预测结果

y_pred_XGB = clf.predict(valid)
y_pred_XGB=pd.DataFrame(y_pred_XGB)   # 转化为DataFrame形式
test_Target=pd.DataFrame(y_valid)

print(accuracy_score(test_Target,y_pred_XGB)) # 评估
print(classification_report(test_Target,y_pred_XGB))

import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('D:/somefilelist/tuite_train.csv')
Target = pd.read_csv('D:/somefilelist/tuite_Target.csv')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Split the dataset into training and testing sets
X_train, X_test = training,valid
y_train, y_test = y_train,y_valid

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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate a random dataset


# Split the dataset into training and testing sets
X_train, X_test = training,valid
y_train, y_test = y_train,y_valid

# Train a random forest classifier with different number of trees
oob_errors = []
for n_trees in range(1,500,50):
    clf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, random_state=42)
    clf.fit(X_train, y_train)
    oob_errors.append(1 - clf.oob_score_)

# Plot the OOB error rate as a function of the number of trees
plt.plot(range(1,500,50), oob_errors)
plt.xlabel('Number of trees')
plt.ylabel('OOB error rate')
plt.show()

oob_errors








