import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import numpy 

import pandas as pd
import pandas

raw_input = pandas.read.csv(open('x.csv'))

raw_input_train, raw_input_test = train_test_split(raw_input,test_size=0.1)

[sampleID_train,glassthickness_train,parameter_train,haze_train,totaltr_train,Rs_train]=np.hsplit(raw_input_train,[1,2,99,100,101,102])

[sampleID_test,glassthickness_test,parameter_test,haze_test,totaltr_test,Rs_test]=np.hsplit(raw_input_test,[1,2,99,100,101,102]

# xgboostモデルの作成
reg1 = xgb.XGBRegressor()
reg2 = xgb.XGBRegressor()
reg3 = xgb.XGBRegressor()

# ハイパーパラメータ探索
reg1_cv = GridSearchCV(reg1, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
reg_cv.fit(parameter_train,haze_train)
print reg1_cv.best_params_, reg1_cv.best_score_

# ハイパーパラメータ探索
reg2_cv = GridSearchCV(reg2, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
reg2_cv.fit(parameter_train,totaltr_train)
print reg2_cv.best_params_, reg2_cv.best_score_

# ハイパーパラメータ探索
reg3_cv = GridSearchCV(reg3, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
reg3_cv.fit(parameter_train,Rs_train)
print reg3_cv.best_params_, reg3_cv.best_score_

# 改めて最適パラメータで学習
reg1 = xgb.XGBRegressor(**reg1_cv.best_params_)
reg1.fit(parameter_train, haze_train)

# 改めて最適パラメータで学習
reg2 = xgb.XGBRegressor(**reg2_cv.best_params_)
reg2.fit(parameter_train,totaltr_train)

# 改めて最適パラメータで学習
reg3 = xgb.XGBRegressor(**reg3_cv.best_params_)
reg3.fit(parameter_train,Rs_train)

# 学習モデルの保存、読み込み
# import pickle
# pickle.dump(reg, open("model.pkl", "wb"))
# reg = pickle.load(open("model.pkl", "rb"))

# 学習モデルの評価
pred1_train = reg1.predict(paramater_train)
pred1_test = reg1.predict(parameter_test)
print mean_squared_error(haze_train, pred1_train)
print mean_squared_error(haze_test, pred1_test)


# feature importance のプロット
import pandas as pd
import matplotlib.pyplot as plt
importances = pd.Series(reg.feature_importances_)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()
