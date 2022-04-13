import scipy
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from matplotlib import pyplot
from sklearn.ensemble import BaggingRegressor
from sklearn import tree
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
import numpy as np


dia = datasets.fetch_openml(data_id=688)

#information of the dataset
# dia.data.info()
# print ("north", dia.data["northing"].unique())
# print("east", dia.data["easting"].unique())
# print ("res", dia.data["resistivity"].unique())
# print ("ins", dia.data["isns"].unique())

#one hot encoder
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), [3])], remainder="passthrough")
updated_data = ct.fit_transform(dia.data)
features = ct.get_feature_names_out()
# print(features)

#panda data
pd_new_data = pd.DataFrame(updated_data, columns = features, index = dia.data.index)

#linear regressor
lr = LinearRegression()
scores = cross_validate(lr, pd_new_data, dia.target, cv=10, scoring="neg_root_mean_squared_error")
rmse = 0-scores["test_score"]
print ("RMSE after cross-validation", rmse)
mean = rmse.mean()
print("RMSE mean", mean)

train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(lr, pd_new_data, dia.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
print ("Curving", test_scores)
rmse = 0-test_scores
print("After curving RMSE", rmse)
rmse_lr = rmse.mean(axis=1)
print ("RMSE Mean after Curving", rmse.mean(axis=1))
print ("train_sizes", train_sizes)
print ("Fit time", np.mean(fit_times[0]))
print ("Score time", score_times)

#Bagging Regressor
br = BaggingRegressor()
train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(br, pd_new_data, dia.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
print ("Test scores", test_scores)
rmse = 0-test_scores
print("RMSE after curving", rmse)
rmse_br = rmse.mean(axis=1)
print ("RMSE Mean after curving", mean)
print ("Train size", train_sizes)
print ("Fit time", fit_times)
print ("Score time", score_times)
print(scipy.stats.ttest_rel(rmse_br, rmse_br))


# #DT Regressor
# dtc = tree.DecisionTreeRegressor(min_samples_leaf=10)
# parameters = [{"min_samples_leaf":[2,4,6,8,10]}]
# tuned_dtc = model_selection.GridSearchCV(dtc, parameters)

# train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(tuned_dtc, pd_new_data, dia.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1],cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
# print ("Test scores", test_scores)
# rmse = 0-test_scores
# print("RMSE", rmse)
# print ("RMSE mean", rmse.mean(axis=1))
# print ("Train size", train_sizes)
# print ("Fit time", fit_times)
# print ("Score time", score_times)

# #Knearest 
# kkparameters = [{"n_neighbors":[2,4,6,8,10]}]
# ktc = KNeighborsRegressor()
# tuned_ktc = model_selection.GridSearchCV(ktc, kkparameters)
# train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(tuned_ktc, pd_new_data, dia.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1],cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)

# print ("Test scores", test_scores)
# rmse = 0-test_scores
# print("RMSE", rmse)
# print ("RMSE Mean", rmse.mean(axis=1))
# print ("train_sizes", train_sizes)
# print ("fit_times", fit_times)
# print ("score_times KN", score_times)

# #svm regressor
# SVRData = SVR()
# train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(SVRData, pd_new_data, dia.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
# print ("Test scores", test_scores)
# rmse = 0-test_scores
# print("RMSE ", rmse)
# rmse.mean(axis=1)
# print ("RMSE Mean", rmse.mean(axis=1))
# print ("train_sizes", train_sizes)
# print ("fit_times", fit_times)
# print ("score_times SV", score_times)


# #dummy_regressor
# DummyData = DummyRegressor()
# train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(DummyData, pd_new_data, dia.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
# print ("Test scores", test_scores)
# rmse = 0-test_scores
# print("RMSE after curving", rmse)
# print ("RMSE mean after curving", rmse.mean(axis=1))
# print ("Train size", train_sizes)
# print ("Fit time", fit_times)
# print ("Score time DM", score_times)