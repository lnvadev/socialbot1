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

train_c.drop(['账号是否为同一天创建','地理定位是否异常',
              '默认配置文件状况是否异常','默认图像配置文件是否异常'],axis=1,inplace=True)

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

X_train = train_c.iloc[:,:-1]
Y_train = train_c['account_type']

training,valid,y_train,y_valid = train_test_split(X_train,Y_train,train_size=0.90,random_state=29)

for i in range(7,17,1):
    clf = RandomForestClassifier(max_features=i,n_estimators=50,random_state=29)
    clf.fit(training, y_train)
    print('max_features' , i , ':')
    print(clf.score(valid,y_valid))

list1 = [100,200,300,400,500]
for i in list1:
    clf = RandomForestClassifier(max_features=10,n_estimators=i,random_state=29)
    clf.fit(training, y_train)
    print('n_estimators' , i , ':')
    print(clf.score(valid,y_valid))

# 决策树
from sklearn import tree
for i in range(3,11,1):
    clf = tree.DecisionTreeClassifier(criterion='gini', max_features=i,max_depth=5,random_state=29)
    clf.fit(training, y_train)
    print('max_features' , i , ':')
    print(round(accuracy_score(clf.predict(valid), y_valid),5))
from sklearn import tree
list1 = [5,10,15,20,25]
for i in list1:
    clf = tree.DecisionTreeClassifier(criterion='gini', max_features=7,max_depth=i,random_state=29)
    clf.fit(training, y_train)
    print('max_features' , i , ':')
    print(round(accuracy_score(clf.predict(valid), y_valid),5))

#载入BP神经网络
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


x_data = X_train
y_data = Y_train
#在X中原来的数值范围是0-225之间，归一化后变为0-1之间
x_data -= x_data.min()
x_data/=x_data.max()-x_data.min()
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.90,random_state=29)
#定义神经网络模型，模型输入神经元个数和输出神经元个数不需要设置
#hidden_layer_sizes用于设置隐藏层结构：
#比如(50)表示有1个隐藏层，隐藏层神经元个数为50
#比如(100,20)表示有2个隐藏层，第1个隐藏层有100个神经元，第2个隐藏层有20个神经元
#比如(100,20,10)表示3个隐藏层，神经元个数分别为100,20,10
#max_iter设置训练次数
list1 = [40,50,60,70]
for i in list1:
    mlp = MLPClassifier(hidden_layer_sizes=[i],learning_rate_init=0.001,max_iter=100,random_state=29)
    mlp.fit(x_train,y_train)
    predictions = mlp.predict(x_test)
    print('============================================',i)
    print(classification_report(y_test,predictions,digits=5))

list1 = [0.001,0.01,0.05,0.1,0.5]
for i in list1:
    mlp = MLPClassifier(hidden_layer_sizes=[50],learning_rate_init=i,max_iter=100,random_state=29)
    mlp.fit(x_train,y_train)
    predictions = mlp.predict(x_test)
    print('============================================',i)
    print(classification_report(y_test,predictions,digits=5))

list1 = [150,200,250,300]
for i in list1:
    mlp = MLPClassifier(hidden_layer_sizes=[50],learning_rate_init=0.01,max_iter=i,random_state=29)
    mlp.fit(x_train,y_train)
    predictions = mlp.predict(x_test)
    print('============================================',i)
    print(classification_report(y_test,predictions,digits=5))

list1 = [0.8945,
0.8997,
0.8967,
0.9038,
0.9043,
0.9019,
0.8923,
0.9021,
0.9022,
0.8959

]

sum(list1)/10.0
import numpy as np
np.std(list1,ddof=1)
precision = 0.8981
recall = 0.8993
print((2*precision*recall)/(precision+recall))