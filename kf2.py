from sklearn.svm import SVR
from sklearn.model_selection import  KFold, cross_val_score
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


data_path = "data/data.csv" # 1600+
# data_path = "data/data_final.csv" # 430

data = pd.read_csv(data_path)
X, y = data.drop('score', axis=1).values, data['score'].values
nf= 10
k_fold = KFold(n_splits=nf, shuffle=True)

reg = GradientBoostingRegressor(n_estimators=300, max_depth=7, learning_rate=0.099, loss='ls')
r2_scores = cross_val_score(reg, X, y, scoring='r2', cv=k_fold)
evc_scores = cross_val_score(reg, X, y, scoring='explained_variance', cv=k_fold)
nmse_scores = cross_val_score(reg, X, y, scoring='neg_mean_squared_error', cv=k_fold)
nmae_scores = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error', cv=k_fold)

print(f"dataset: {X.shape}; {nf}-Fold")

print(f"Reg: {str(reg).split('(')[0]}")

print(f"R2: {r2_scores}\n"
      f"mean: {r2_scores.mean()}\t"
      f"max: {r2_scores.max()}\t"
      f"min: {r2_scores.min()}\n")

print(f"EVC: {evc_scores}\n"
      f"mean: {evc_scores.mean()}\t"
      f"max: {evc_scores.max()}\t"
      f"min: {evc_scores.min()}\n ")


print(f"NMSE: {nmse_scores}\n"
      f"mean: {nmse_scores.mean()}\t"
      f"max: {nmse_scores.max()}\t"
      f"min: {nmse_scores.min()}\n ")

print(f"NMAE: {nmae_scores}\n"
      f"mean: {nmae_scores.mean()}\t"
      f"max: {nmae_scores.max()}\t"
      f"min: {nmae_scores.min()}\n ")

print("# ____________________________________________________________________ #")

reg = XGBRegressor(n_estimators=700, objective="reg:squarederror", max_depth=7, learning_rate=0.099)
r2_scores = cross_val_score(reg, X, y, scoring='r2', cv=k_fold)
evc_scores = cross_val_score(reg, X, y, scoring='explained_variance', cv=k_fold)
nmse_scores = cross_val_score(reg, X, y, scoring='neg_mean_squared_error', cv=k_fold)
nmae_scores = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error', cv=k_fold)

print(f"dataset: {X.shape}; {nf}-Fold")

print(f"Reg: {str(reg).split('(')[0]}")

print(f"R2: {r2_scores}\n"
      f"mean: {r2_scores.mean()}\t"
      f"max: {r2_scores.max()}\t"
      f"min: {r2_scores.min()}\n")

print(f"EVC: {evc_scores}\n"
      f"mean: {evc_scores.mean()}\t"
      f"max: {evc_scores.max()}\t"
      f"min: {evc_scores.min()}\n ")


print(f"NMSE: {nmse_scores}\n"
      f"mean: {nmse_scores.mean()}\t"
      f"max: {nmse_scores.max()}\t"
      f"min: {nmse_scores.min()}\n ")

print(f"NMAE: {nmae_scores}\n"
      f"mean: {nmae_scores.mean()}\t"
      f"max: {nmae_scores.max()}\t"
      f"min: {nmae_scores.min()}\n ")
print("# ____________________________________________________________________ #")
reg = MLPRegressor(hidden_layer_sizes=(250,),learning_rate='adaptive', random_state=0)
r2_scores = cross_val_score(reg, X, y, scoring='r2', cv=k_fold)
evc_scores = cross_val_score(reg, X, y, scoring='explained_variance', cv=k_fold)
nmse_scores = cross_val_score(reg, X, y, scoring='neg_mean_squared_error', cv=k_fold)
nmae_scores = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error', cv=k_fold)

print(f"dataset: {X.shape}; {nf}-Fold")

print(f"Reg: {str(reg).split('(')[0]}")

print(f"R2: {r2_scores}\n"
      f"mean: {r2_scores.mean()}\t"
      f"max: {r2_scores.max()}\t"
      f"min: {r2_scores.min()}\n")

print(f"EVC: {evc_scores}\n"
      f"mean: {evc_scores.mean()}\t"
      f"max: {evc_scores.max()}\t"
      f"min: {evc_scores.min()}\n ")


print(f"NMSE: {nmse_scores}\n"
      f"mean: {nmse_scores.mean()}\t"
      f"max: {nmse_scores.max()}\t"
      f"min: {nmse_scores.min()}\n ")

print(f"NMAE: {nmae_scores}\n"
      f"mean: {nmae_scores.mean()}\t"
      f"max: {nmae_scores.max()}\t"
      f"min: {nmae_scores.min()}\n ")
print("# ____________________________________________________________________ #")
reg = SVR(kernel='rbf', C=25, gamma=0.1, epsilon=.1)
r2_scores = cross_val_score(reg, X, y, scoring='r2', cv=k_fold)
evc_scores = cross_val_score(reg, X, y, scoring='explained_variance', cv=k_fold)
nmse_scores = cross_val_score(reg, X, y, scoring='neg_mean_squared_error', cv=k_fold)
nmae_scores = cross_val_score(reg, X, y, scoring='neg_mean_absolute_error', cv=k_fold)

print(f"dataset: {X.shape}; {nf}-Fold")

print(f"Reg: {str(reg).split('(')[0]}")

print(f"R2: {r2_scores}\n"
      f"mean: {r2_scores.mean()}\t"
      f"max: {r2_scores.max()}\t"
      f"min: {r2_scores.min()}\n")

print(f"EVC: {evc_scores}\n"
      f"mean: {evc_scores.mean()}\t"
      f"max: {evc_scores.max()}\t"
      f"min: {evc_scores.min()}\n ")


print(f"NMSE: {nmse_scores}\n"
      f"mean: {nmse_scores.mean()}\t"
      f"max: {nmse_scores.max()}\t"
      f"min: {nmse_scores.min()}\n ")

print(f"NMAE: {nmae_scores}\n"
      f"mean: {nmae_scores.mean()}\t"
      f"max: {nmae_scores.max()}\t"
      f"min: {nmae_scores.min()}\n ")

# ____________________________________________________________________ #


