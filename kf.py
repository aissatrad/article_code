from sklearn.svm import SVR
from sklearn.model_selection import  KFold, cross_validate
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


data_path = "data/data.csv" # 1600+
# data_path = "data/data_final.csv" # 430

data = pd.read_csv(data_path)
X, y = data.drop('score', axis=1).values, data['score'].values
k_fold = KFold(n_splits=10, shuffle=True)

reg = GradientBoostingRegressor(n_estimators=300, max_depth=7, learning_rate=0.099, loss='ls')

scores = cross_validate(reg, X, y, scoring=['explained_variance',
                                            'r2', 'neg_mean_squared_error',
                                            'neg_mean_absolute_error'], cv=k_fold)
# print(f"dataset: {X.shape}; 5-Fold")
print(f"Reg: {str(reg).split('(')[0]}")
print(scores)
print(f"R2: {scores.get('test_r2')}\n"
      f"EVC: {scores.get('test_explained_variance')}\n"
      f"NMSE: {scores.get('test_neg_mean_squared_error')}\n"
      f"NMAE: {scores.get('test_neg_mean_absolute_error')} ")

print("# ____________________________________________________________________ #")

reg = XGBRegressor(n_estimators=700, objective="reg:squarederror", max_depth=7, learning_rate=0.099)
scores = cross_validate(reg, X, y, scoring=['explained_variance',
                                            'r2', 'neg_mean_squared_error',
                                            'neg_mean_absolute_error'], cv=k_fold)
# print(f"dataset: {X.shape}; 5-Fold")
print(f"Reg: {str(reg).split('(')[0]}")
print(f"R2: {scores.get('test_r2')}\n"
      f"EVC: {scores.get('test_explained_variance')}\n"
      f"MSE: {scores.get('test_neg_mean_squared_error')}\n"
      f"MAE: {scores.get('test_neg_mean_absolute_error')} ")
print("# ____________________________________________________________________ #")
reg = MLPRegressor(hidden_layer_sizes=(250,),learning_rate='adaptive', random_state=0)
scores = cross_validate(reg, X, y, scoring=['explained_variance',
                                            'r2', 'neg_mean_squared_error',
                                            'neg_mean_absolute_error'], cv=k_fold)
# print(f"dataset: {X.shape}; 5-Fold")
print(f"Reg: {str(reg).split('(')[0]}")
print(f"R2: {scores.get('test_r2')}\n"
      f"EVC: {scores.get('test_explained_variance')}\n"
      f"MSE: {scores.get('test_neg_mean_squared_error')}\n"
      f"MAE: {scores.get('test_neg_mean_absolute_error')} ")

print("# ____________________________________________________________________ #")
reg = SVR(kernel='rbf', C=25, gamma=0.1, epsilon=.1)
scores = cross_validate(reg, X, y, scoring=['explained_variance',
                                            'r2', 'neg_mean_squared_error',
                                            'neg_mean_absolute_error'], cv=k_fold)
# print(f"dataset: {X.shape}; 5-Fold")
print(f"Reg: {str(reg).split('(')[0]}")
print(f"R2: {scores.get('test_r2')}\n"
      f"EVC: {scores.get('test_explained_variance')}\n"
      f"MSE: {scores.get('test_neg_mean_squared_error')}\n"
      f"MAE: {scores.get('test_neg_mean_absolute_error')} ")

# ____________________________________________________________________ #


