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

import warnings 
warnings.filterwarnings("ignore")

# 随机森林
X_train = train_c.iloc[:,:-1]
Y_train = train_c['account_type']
list1 = [11,12,13,14,15,16,17,18,19,20]
# 准确率
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
for i in list1:
    training,valid,y_train,y_valid = train_test_split(X_train,Y_train,train_size=0.90,random_state=i)
    kfold = KFold(n_splits=10, shuffle=True, random_state=29)
    model= RandomForestClassifier(max_features=10,n_estimators=400,random_state=29) 
    model.fit(training, y_train)
    Predict = model.predict(valid)
    print('=======================',i)
    print(accuracy_score(Predict,y_valid))
    print(precision_score(Predict,y_valid, average='macro'))
    print(recall_score(Predict,y_valid, average='macro'))

# 决策树
# 准确率
from sklearn import tree
kfold = KFold(n_splits=10, shuffle=True, random_state=29)
model= tree.DecisionTreeClassifier(criterion='gini', max_features=7,max_depth=15,random_state=29)
scores = cross_validate(estimator=model,X=X_train,y=Y_train,
                        cv=kfold)
print(scores)

scoring = ['precision_macro', 'recall_macro']
kfold = KFold(n_splits=10, shuffle=True, random_state=29)
model= tree.DecisionTreeClassifier(criterion='gini', max_features=7,max_depth=15,random_state=29)
scores = cross_validate(estimator=model,X=X_train,y=Y_train,
                        cv=kfold,scoring=scoring)
print(scores)

# 神经网络
from sklearn.neural_network import MLPClassifier
x_data = X_train
y_data = Y_train
#在X中原来的数值范围是0-225之间，归一化后变为0-1之间
x_data -= x_data.min()
x_data/=x_data.max()-x_data.min()

kfold = KFold(n_splits=10, shuffle=True, random_state=29)
model= MLPClassifier(hidden_layer_sizes=[50],learning_rate_init=0.01,max_iter=200,random_state=29)
scores = cross_validate(estimator=model,X=x_data,y=y_data,
                        cv=kfold)
print(scores)


scoring = ['precision_macro', 'recall_macro']
kfold = KFold(n_splits=10, shuffle=True, random_state=29)
model= MLPClassifier(hidden_layer_sizes=[50],learning_rate_init=0.01,max_iter=200,random_state=29)
scores = cross_validate(estimator=model,X=x_data,y=y_data,
                        cv=kfold,scoring=scoring)
print(scores)

# 基学习器
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

train_c.drop(['关注数与被关注数的比值','点赞数与关注数的比值',
              '发文总量与关注数的比值','账号描述是否异常',
             '点赞数与发文总量比值','粉丝数量与累计登录天数的比值',
              '点赞数与累计登录天数的比值','关注数与累计登录天数的比值','发文总数与粉丝数量比值'],axis=1,inplace=True)

train_c.info()

X_train = train_c.iloc[:,:-1]
Y_train = train_c['account_type']
list1 = [11,12,13,14,15,16,17,18,19,20]






