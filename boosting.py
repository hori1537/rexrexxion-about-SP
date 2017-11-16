import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error


# データ読み込み
boston = load_boston()
X_train, X_test = boston.data[:400], boston.data[400:]
y_train, y_test = boston.target[:400], boston.target[400:]

# xgboostモデルの作成
reg = xgb.XGBRegressor()

# ハイパーパラメータ探索
reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
reg_cv.fit(X_train, y_train)
print reg_cv.best_params_, reg_cv.best_score_

# 改めて最適パラメータで学習
reg = xgb.XGBRegressor(**reg_cv.best_params_)
reg.fit(X_train, y_train)

# 学習モデルの保存、読み込み
# import pickle
# pickle.dump(reg, open("model.pkl", "wb"))
# reg = pickle.load(open("model.pkl", "rb"))

# 学習モデルの評価
pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)
print mean_squared_error(y_train, pred_train)
print mean_squared_error(y_test, pred_test)

# feature importance のプロット
import pandas as pd
import matplotlib.pyplot as plt
importances = pd.Series(reg.feature_importances_, index = boston.feature_names)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()
