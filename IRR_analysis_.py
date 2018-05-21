# -*- coding: utf-8 -*-

###各種ライブラリをインポート
import xgboost as xgb
from xgboost import plot_tree

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
#ver0.20
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler

import numpy as np
import numpy 
import pandas as pd
import pandas

import lime
import lime.lime_tabular

import sys

import matplotlib.pyplot as plt

import math

from inspect import currentframe

def chkprint(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))


###フォルダのアドレス定義
address_ = 'D:/xgboost/'
#address_ = 'C:/Users/1310202/Documents/xgboost/'
#address_ = '/home/garaken/xgboost/'

###乱数を固定（seed(1)）
np.random.seed(1)


#df_raw_input = pandas.read_csv(open('D:/xgboost/VISS_IRR_utf8.csv'))
#df_raw_input = pandas.read_csv(open('C:/Users/1310202/Documents/xgboost/VISS_IRR_utf8.csv' ,encoding="utf-8_sig"))

###フォルダのアドレス　+ 'VISS_IRR_utf8-.csv'  を読み込み
df_raw_input = pandas.read_csv(open(str(address_) + 'VISS_IRR_utf8-.csv' ,encoding="utf-8_sig"))

###要素名の取得（例：SiAlN thickness, SiAlN power...）
df_features = df_raw_input.iloc[:,2:106]
feature_names = df_features.columns
chkprint(feature_names)

###train_test_split を用いて、訓練データとテストデータに分割
raw_input_train, raw_input_test = train_test_split(df_raw_input,test_size=0.1)

###pandasをnumpyに変換
raw_input_train = np.array(raw_input_train)
raw_input_test = np.array(raw_input_test)



#dataset
[sampleID_train, glassthickness_train, parameter_train, haze_train,tr_train,td_train,tp_train,Rs_train]=np.hsplit(raw_input_train, [1,2,105,106,107,108,109])
[sampleID_test, glassthickness_test, parameter_test, haze_test,tr_test,td_test,tp_test,Rs_test]=np.hsplit(raw_input_test, [1,2,105,106,107,108,109])

haze_train = np.log10(haze_train+0.00001)
haze_test = np.log10(haze_test+0.00001)


print('##################### StandardScaler #####################')
std_scl = StandardScaler()
std_scl.fit(parameter_train)

parameter_train = std_scl.transform(parameter_train)
parameter_test = std_scl.transform(parameter_test)

'''
parameter_train = std_scl.inverse_transform(parameter_train)
parameter_test = std_scl.inverse_transform(parameter_test)
'''


print('##################### Regression of Stochastic Gradient Descent #####################')
reg1_SGD = linear_model.SGDRegressor(max_iter=1000)
reg2_SGD = linear_model.SGDRegressor(max_iter=1000)
ref3_SGD = linear_model.SGDRegressor(max_iter=1000)

reg1_SGD.fit(parameter_train, haze_train)
reg2_SGD.fit(parameter_train, Rs_train)
#reg3_SGD.fit(parameter_train, tr_train)


'''
print("SGD_haze")
print(reg1_SGD.intercept_) 
print(reg1_SGD.coef_) 

print("SGD_Rs")
print(reg2_SGD.intercept_) 
print(reg2_SGD.coef_) 
'''

print('SGD haze mean_squared_error')
print(sklearn.metrics.mean_squared_error(haze_test, reg1_SGD.predict(parameter_test)))

print('SGD rs mean_squared_error')
print(sklearn.metrics.mean_squared_error(Rs_test, reg2_SGD.predict(parameter_test)))


print('##################### Regression of Ridge #####################')
reg1_Ridge = linear_model.Ridge(alpha=1.0)
reg2_Ridge = linear_model.Ridge(alpha=1.0)
#reg3_Ridge = linear_model.Ridge(alpha=1.0)

reg1_Ridge.fit(parameter_train,haze_train)
reg2_Ridge.fit(parameter_train,Rs_train)

'''
print("Ridge_haze")
print(reg1_Ridge.intercept_) 
print(reg1_Ridge.coef_) 

print("Ridge_Rs")
print(reg2_Ridge.intercept_) 
print(reg2_Ridge.coef_) 
'''

print('Ridge haze mean_squared_error')
print(sklearn.metrics.mean_squared_error(haze_test, reg1_Ridge.predict(parameter_test)))

print('Ridge Rs mean_squared_error')
print(sklearn.metrics.mean_squared_error(Rs_test, reg2_Ridge.predict(parameter_test)))


print('##################### Regression of Lasso #####################')
reg1_Lasso = linear_model.Lasso(alpha=1.0)
reg2_Lasso = linear_model.Lasso(alpha=1.0)
#reg3_Lasso = linear_model.Lasso(alpha=1.0)

reg1_Lasso.fit(parameter_train,haze_train)
reg2_Lasso.fit(parameter_train,Rs_train)


'''
print("Lasso_haze")
print(reg1_Lasso.intercept_) 
print(reg1_Lasso.coef_) 

print("Lasso_Rs")
print(reg2_Lasso.intercept_) 
print(reg2_Lasso.coef_) 
'''

print('Lasso haze mean_squared_error')
print(sklearn.metrics.mean_squared_error(haze_test, reg1_Lasso.predict(parameter_test)))

print('Lasso Rs mean_squared_error')
print(sklearn.metrics.mean_squared_error(Rs_test, reg2_Lasso.predict(parameter_test)))


print('##################### Regression of Elastic Net #####################')
reg1_ElasticNet = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
reg2_ElasticNet = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
#reg3_ElasticNet = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)

reg1_ElasticNet.fit(parameter_train,haze_train)
reg2_ElasticNet.fit(parameter_train,Rs_train)


'''
print("ElasticNet_haze")
print(reg1_ElasticNet.intercept_) 
print(reg1_ElasticNet.coef_) 

print("ElasticNet_Rs")
print(reg2_ElasticNet.intercept_) 
print(reg2_ElasticNet.coef_) 
'''

print('ElasticNet haze mean_squared_error')
print(sklearn.metrics.mean_squared_error(haze_test, reg1_ElasticNet.predict(parameter_test)))

print('ElasticNet Rs mean_squared_error')
print(sklearn.metrics.mean_squared_error(Rs_test, reg2_ElasticNet.predict(parameter_test)))


print('##################### Regression of SVR #####################')
reg1_SVR = svm.SVR(kernel='rbf', C=1)
reg2_SVR = svm.SVR(kernel='rbf', C=1)
ref3_SVR = svm.SVR(kernel='rbf', C=1)

reg1_SVR.fit(parameter_train,haze_train)
reg2_SVR.fit(parameter_train,Rs_train)
#reg3_SVR.fit(parameter_train,tr_train)


print('SVR haze mean_squared_error')
print(sklearn.metrics.mean_squared_error(haze_test, reg1_SVR.predict(parameter_test)))

print('SVR Rs mean_squared_error')
print(sklearn.metrics.mean_squared_error(Rs_test, reg2_SVR.predict(parameter_test)))


haze_pred_SVR = reg1_SVR.predict(parameter_test)
tr_pred_SVR = reg2_SVR.predict(parameter_test)

haze_pred_SVR = reg1_SVR.predict(parameter_test)
tr_pred_SVR = reg2_SVR.predict(parameter_test)


print('##################### Regression of DecisionTreeRegressor #####################')
reg1_DTR = sklearn.tree.DecisionTreeRegressor(max_depth =4)
reg2_DTR = sklearn.tree.DecisionTreeRegressor(max_depth =4)
ref3_DTR = sklearn.tree.DecisionTreeRegressor(max_depth =4)

reg1_DTR.fit(parameter_train,haze_train)
reg2_DTR.fit(parameter_train,Rs_train)
#reg3_DTR.fit(parameter_train,tr_train)


print('DTR haze mean_squared_error')
print(sklearn.metrics.mean_squared_error(haze_test, reg1_DTR.predict(parameter_test)))

print('DTR Rs mean_squared_error')
print(sklearn.metrics.mean_squared_error(Rs_test, reg2_DTR.predict(parameter_test)))


haze_pred_DTR = reg1_SGD.predict(parameter_test)
tr_pred_DTR = reg2_SGD.predict(parameter_test)

haze_pred_DTR = reg1_DTR.predict(parameter_test)
tr_pred_DTR = reg2_DTR.predict(parameter_test)


print('##################### Regression of RandomForestRegressor #####################')
reg1_RFR = ensemble.RandomForestRegressor()
reg2_RFR = ensemble.RandomForestRegressor()
ref3_RFR = ensemble.RandomForestRegressor()

reg1_RFR.fit(parameter_train,haze_train)
reg2_RFR.fit(parameter_train,Rs_train)
#reg3_RFR.fit(parameter_train,tr_train)


print('RFR haze mean_squared_error')
print(sklearn.metrics.mean_squared_error(haze_test, reg1_RFR.predict(parameter_test)))

print('RFR Rs mean_squared_error')
print(sklearn.metrics.mean_squared_error(Rs_test, reg2_RFR.predict(parameter_test)))


haze_pred_RFR = reg1_RFR.predict(parameter_test)
tr_pred_RFR = reg2_RFR.predict(parameter_test)

haze_pred_RFR = reg1_RFR.predict(parameter_test)
tr_pred_RFR = reg2_RFR.predict(parameter_test)


print('##################### Regression of XGBoost #####################')

try:

	reg1_gbtree= xgb.XGBRegressor()
	reg2_gbtree= xgb.XGBRegressor()
	reg3_gbtree= xgb.XGBRegressor()

	reg1_gbtree_cv = GridSearchCV(reg1_gbtree, {'max_depth': [3,4,5], 'n_estimators': [50,100,200,250]}, verbose=1)
	reg1_gbtree_cv.fit(parameter_train,haze_train)
	print('best gbregressor ', reg1_gbtree_cv.best_params_, reg1_gbtree_cv.best_score_)

	reg2_gbtree_cv = GridSearchCV(reg2_gbtree, {'max_depth': [3,4,5], 'n_estimators': [50,100,200,250]}, verbose=1)
	reg2_gbtree_cv.fit(parameter_train,Rs_train)
	
	print(reg2_gbtree_cv.best_params_, reg2_gbtree_cv.best_score_)

	#reg3_gbtree_cv = GridSearchCV(reg3_gbtree, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
	#reg3_gbtree_cv.fit(parameter_train,Rs_train)
	#print(reg3_gbtree_cv.best_params_, reg3_cv.best_score_)

	reg1_gbtree= xgb.XGBRegressor(**reg1_gbtree_cv.best_params_)
	reg1_gbtree.fit(parameter_train, haze_train)

	reg2_gbtree= xgb.XGBRegressor(**reg2_gbtree_cv.best_params_)
	reg2_gbtree.fit(parameter_train,Rs_train)

	#reg3_gbtree= xgb.XGBRegressor(**reg3_gbtree_cv.best_params_)
	#reg3_gbtree.fit(parameter_train,tr_train)
	
	print('XGBoost haze mean_squared_error')
	print(sklearn.metrics.mean_squared_error(haze_test, reg1_gbtree.predict(parameter_test)))

	print('XGBoost Rs mean_squared_error')
	print(sklearn.metrics.mean_squared_error(Rs_test, reg2_gbtree.predict(parameter_test)))


except:
	print('Cant excute XGBOOST')


print('##################### Keras LSTM #####################')

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation ='sigmoid'))

model.compile(loss='mse',
		optimizer='rmsprop'
		metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16, epochs=100)

score =model.evaluate(x_test,y_test,batch_size=16)





	
print('##################### Save as CSV file #####################')
df_SGD_haze_intercept = pd.DataFrame(reg1_SGD.intercept_)
df_SGD_haze_coef = pd.DataFrame(reg1_SGD.coef_)
df_SGD_haze =pd.concat([df_SGD_haze_intercept,df_SGD_haze_coef])

df_SGD_Rs_intercept = pd.DataFrame(reg2_SGD.intercept_)
df_SGD_Rs_coef = pd.DataFrame(reg2_SGD.coef_)
df_SGD_Rs =pd.concat([df_SGD_Rs_intercept,df_SGD_Rs_coef])

df_SGD_haze.to_csv(str(address_) + 'SGD_haze.csv')
df_SGD_Rs.to_csv(str(address_) + 'SGD_Rs.csv')

df_Ridge_haze_intercept = pd.DataFrame(reg1_Ridge.intercept_)
df_Ridge_haze_coef = pd.DataFrame(reg1_Ridge.coef_)
df_Ridge_haze =pd.concat([df_Ridge_haze_intercept,df_Ridge_haze_coef])

df_Ridge_Rs_intercept = pd.DataFrame(reg2_Ridge.intercept_)
df_Ridge_Rs_coef = pd.DataFrame(reg2_Ridge.coef_)
df_Ridge_Rs =pd.concat([df_Ridge_Rs_intercept,df_Ridge_Rs_coef])

df_Ridge_haze.to_csv(str(address_) + 'Ridge_haze.csv')
df_Ridge_Rs.to_csv(str(address_) + 'Ridge_Rs.csv')


print('##################### LIME Explainer #####################')
#try:
#explainer1 = lime.lime_tabular.LimeTabularExplainer(haze_train, feature_names=feature_names, kernel_width=3)
#	explainer1 = lime.lime_tabular.LimeTabularExplainer(parameter_train, feature_names=feature_names, class_names=['haze'], verbose=True, mode='regression')
explainer1 = lime.lime_tabular.LimeTabularExplainer(parameter_train, feature_names=feature_names, class_names=['haze'], verbose=True, mode='regression')

np.random.seed(1)
i = 3
exp = explainer1.explain_instance(parameter_test[i], reg1_SVR.predict, num_features=5)

# exp.show_in_notebook(show_all=False)
exp.save_to_file(file_path= str(address_) + 'numeric_category_feat_01', show_table=True, show_all=True)

i = 3
exp = explainer1.explain_instance(parameter_test[i], reg1_SVR.predict, num_features=10)
# exp.show_in_notebook(show_all=False)
exp.save_to_file(file_path=str(address_) + 'numeric_category_feat_02', show_table=True, show_all=True)
#except:
#print('Cannot excute LIME Explainer')


#####################    to pickel    #####################
# import pickle
# pickle.dump(reg, open("model.pkl", "wb"))
# reg = pickle.load(open("model.pkl", "rb"))

pred1_train = reg1_gbtree.predict(parameter_train)
pred1_test = reg1_gbtree.predict(parameter_test)
print(mean_squared_error(haze_train, pred1_train))
print(mean_squared_error(haze_test, pred1_test))

import matplotlib.pyplot as plt


#####################    plot gbtree    #####################
try:
	importances = pd.Series(reg1_gbtree.feature_importances_)
	importances = importances.sort_values()
	importances.plot(kind = "barh")
	plt.title("imporance in the xgboost Model")
	plt.show()

	importances = pd.Series(reg2_gbtree.feature_importances_)
	importances = importances.sort_values()
	importances.plot(kind = "barh")
	plt.title("imporance in the xgboost Model")
	plt.show()

	#xgb.plot_tree(reg1_gbtree)
	#xgb.to_graphviz(reg2_gbtree, num_trees=1)
	#xgb.plot_tree(reg2_gbtree, num_trees=1)


	#xgb.to_graphviz(reg1_gbtree,num_trees=1)
	#_, axes= plt.subplots(1,5)
	for i in range(4):
	    xgb.plot_tree(reg2_gbtree, num_trees=i)
	plt.show()
	
except:
	print('Cannot plot gbtree')
