import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import *
import time
from sklearn.decomposition import PCA

data_path = "data/final_init.csv"


def prepare_data():
    global x_train, x_test, y_train, y_test
    to_drop = ['score']
    data = pd.read_csv(data_path)
    labels = data.score
    train = data.drop(to_drop, axis=1)
    # pca = PCA(n_components=100, random_state=100)
    # pca = PCA(n_components=55, random_state=100)
    pca = PCA(n_components=15, random_state=100)
    principalComponents = pca.fit_transform(train)
    principalDf = pd.DataFrame(data=principalComponents)

    print(principalDf.shape)
    return train_test_split(principalDf, labels, test_size=0.3, random_state=15)


x_train, x_test, y_train, y_test = prepare_data()

gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=300, max_depth=7, learning_rate=0.099, loss='ls')
start = time.process_time()
gradient_boosting_regressor.fit(x_train, y_train)
end = time.process_time()
print(f"gradient_boosting_regressor: Execution time : {end - start} s")
predictions = gradient_boosting_regressor.predict(x_test)
print(f"explained_variance_score : {explained_variance_score(y_test, predictions)}")
print(f"r2_score : {r2_score(y_test, predictions)}")
print(f"mean_squared_error : {mean_squared_error(y_test, predictions)}")
print(f"mean_tweedie_deviance : {mean_tweedie_deviance(y_test, predictions)}")
print(f"median_absolute_error : {median_absolute_error(y_test, predictions)}")
print(f"max_error : {max_error(y_test, predictions)}")
print(f"mean_absolute_error : {mean_absolute_error(y_test, predictions)}")


print('\n\n')

xgb_regressor = XGBRegressor(n_estimators=700, objective="reg:squarederror", max_depth=7, learning_rate=0.099)
start = time.process_time()
xgb_regressor.fit(x_train, y_train)
end = time.process_time()
print(f"xgb_regressor: Execution time : {end - start} s")
predictions = xgb_regressor.predict(x_test)
print(f"explained_variance_score : {explained_variance_score(y_test, predictions)}")
print(f"r2_score : {r2_score(y_test, predictions)}")
print(f"mean_squared_error : {mean_squared_error(y_test, predictions)}")
print(f"mean_tweedie_deviance : {mean_tweedie_deviance(y_test, predictions)}")
print(f"median_absolute_error : {median_absolute_error(y_test, predictions)}")
print(f"max_error : {max_error(y_test, predictions)}")
print(f"mean_absolute_error : {mean_absolute_error(y_test, predictions)}")


print('\n\n')

sv_regressor = SVR(kernel='rbf', C=25, gamma=0.1, epsilon=.1)
start = time.process_time()
sv_regressor.fit(x_train, y_train)
end = time.process_time()
print(f"sv_regressor: Execution time : {end - start} s")
predictions = sv_regressor.predict(x_test)
print(f"explained_variance_score : {explained_variance_score(y_test, predictions)}")
print(f"r2_score : {r2_score(y_test, predictions)}")
print(f"mean_squared_error : {mean_squared_error(y_test, predictions)}")
print(f"mean_tweedie_deviance : {mean_tweedie_deviance(y_test, predictions)}")
print(f"median_absolute_error : {median_absolute_error(y_test, predictions)}")
print(f"max_error : {max_error(y_test, predictions)}")
print(f"mean_absolute_error : {mean_absolute_error(y_test, predictions)}")


print('\n\n')

regrMLPRegressor = MLPRegressor(hidden_layer_sizes=(250,),learning_rate='adaptive', random_state=0)
start = time.process_time()
regrMLPRegressor.fit(x_train, y_train)
end = time.process_time()
print(f"regrMLPRegressor: Execution time : {end - start} s")
predictions = regrMLPRegressor.predict(x_test)
print(f"explained_variance_score : {explained_variance_score(y_test, predictions)}")
print(f"r2_score : {r2_score(y_test, predictions)}")
print(f"mean_squared_error : {mean_squared_error(y_test, predictions)}")
print(f"mean_tweedie_deviance : {mean_tweedie_deviance(y_test, predictions)}")
print(f"median_absolute_error : {median_absolute_error(y_test, predictions)}")
print(f"max_error : {max_error(y_test, predictions)}")
print(f"mean_absolute_error : {mean_absolute_error(y_test, predictions)}")

model_file = open("model/GBRegressor_final.sav", 'wb')
pickle.dump(gradient_boosting_regressor, model_file)
model_file.close()
