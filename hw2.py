import numpy as np
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
import scipy


def load_dataset():
    dia = datasets.fetch_openml(data_id=688)
    return dia

def dataset_information(dia):   
    dia.data.info()
    print ("north", dia.data["northing"].unique())
    print("east", dia.data["easting"].unique())
    print ("res", dia.data["resistivity"].unique())
    print ("ins", dia.data["isns"].unique())

def one_hot_encoder(dia):
    ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), [3])], remainder="passthrough")
    updated_data = ct.fit_transform(dia.data)
    
    features = ct.get_feature_names_out()
    print(features)
    return updated_data, features

def panda_stuff(ohe_data, features, dia_data):
    pd_new_data = pd.DataFrame(ohe_data, columns = features, index = dia_data.data.index)
    return pd_new_data

def linear_regression(pd_data, dia_data):
    lr = LinearRegression(fit_intercept=True)
    scores = cross_validate(lr, pd_data, dia_data.target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-scores["test_score"]  
    file_write_cross("Linear Regressor", rmse.mean())

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(lr, pd_data, dia_data.target,
     train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",
     shuffle=True,random_state=0)
    rmse = 0-test_scores
    rmse_mean =  rmse.mean(axis=1) 
    file_write_curve("Linear Regressor", rmse_mean, train_sizes, fit_times, score_times)

    plot(train_sizes, rmse_mean, "Number of training examples", "RMSE", "Linear Regressor")
    return rmse


def bagging_regressor(pd_data, dia_data):
    br = BaggingRegressor()   
    bagged_cross = cross_validate(br, pd_data, dia_data.target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-bagged_cross["test_score"]
    file_write_cross("Bagged Regressor", rmse.mean())


    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(br, pd_data, dia_data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
    rmse = 0-test_scores
    rmse_mean =  rmse.mean(axis=1)
    file_write_curve("Bagged Regressor", rmse_mean, train_sizes, fit_times, score_times)

    plot(train_sizes, rmse_mean, "Number of training examples", "RMSE", "Bagged Regressor")
    return rmse


def decission_tree_regressor(pd_data, data):
    dtc = tree.DecisionTreeRegressor(min_samples_leaf=10)  
    parameters = [{"min_samples_leaf":[2,4,6,8,10]}]     
    tuned_dtc = model_selection.GridSearchCV(dtc, parameters)
    dtc_cross = cross_validate(tuned_dtc, pd_data, data.target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-dtc_cross["test_score"]
    file_write_cross("Decision Tree Regressor", rmse.mean()) 
    
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(tuned_dtc, pd_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1],cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
    rmse = 0-test_scores
    rmse_mean =  rmse.mean(axis=1)
    file_write_curve("Decision  Tree Regressor", rmse_mean, train_sizes, fit_times, score_times)

    
    plot(train_sizes, rmse_mean, "Number of training examples", "RMSE", "Decision Tree Regressor")
    return rmse

def KNeighbors_regression(pd_data, data):
    parameters = [{"n_neighbors":[2,4,6,8,10]}]
    knn = KNeighborsRegressor(n_neighbors=5)
    tuned_kn = model_selection.GridSearchCV(knn, parameters)
    knn_cross = cross_validate(tuned_kn, pd_data, data.target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-knn_cross["test_score"]
    file_write_cross("K nearest Neighbor Regressor", rmse.mean())

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(tuned_kn, pd_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1],cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)    
    rmse = 0-test_scores
    rmse_mean =  rmse.mean(axis=1)
    file_write_curve("K Nearest Neighbor Regressor", rmse_mean, train_sizes, fit_times, score_times)
    
    plot(train_sizes, rmse_mean, "Number of training examples", "RMSE", "K Nearest Neighbor Regressor")
    return rmse

def SVM_reggressior(pd_data, data):
    SVRData = SVR()
    svm_cross = cross_validate(SVRData, pd_data, data.target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-svm_cross["test_score"]
    file_write_cross("Dummy Regressor", rmse.mean())
    
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(SVRData, pd_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
    rmse = 0-test_scores
    rmse_mean =  rmse.mean(axis=1)
    file_write_curve("SVM Regressor", rmse_mean, train_sizes, fit_times, score_times)
    
    plot(train_sizes, rmse.mean(axis=1), "Number of training examples", "RMSE", "SVM Regressor")
    return rmse

def dummy_regressor(pd_data, data):
    DummyData = DummyRegressor()
    dummy_cross = cross_validate(DummyData, pd_data, data.target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-dummy_cross["test_score"]
    file_write_cross("Dummy Regressor", rmse.mean())

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(DummyData, pd_data, data.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
    rmse = 0-test_scores
    rmse_mean =   rmse.mean(axis=1)

    file_write_curve("Dummy Regressor", rmse_mean, train_sizes, fit_times, score_times)

    plot(train_sizes, rmse_mean, "Number of training examples", "RMSE", "Dummy Regressor")
    return rmse

def statistical_significance(curving_rmse_1, curving_rmse_2):
    return scipy.stats.ttest_rel(curving_rmse_1[4], curving_rmse_2[4])

def file_write_statistical_significance( method, value):
    textfile = open("HW2_Result_significance.txt", "a")
    textfile.write(method + "\n")
    textfile.write(str(value) + " ")
    textfile.close()

def file_write_cross( method, rmse):
    textfile = open("HW2_Result_cross.txt", "a")
    textfile.write(method + "\n")
    textfile.write(str(rmse) + " ")
    textfile.write("\n")
    textfile.close()

def file_write_curve(method, rmse, train_size, fit_times, score_times):
    textfile = open("HW2_Result_curve.txt", "a")
    textfile.write(method + "\n")
    for i in rmse:
        textfile.write(str(i) + " ")
    textfile.write("\n")

    for i in train_size:
        textfile.write(str(i) + " ")
    textfile.write("\n")

    for i in fit_times:
        textfile.write(str(np.mean(i)) + " ")
    textfile.write("\n")

    for i in score_times:
        textfile.write(str(np.mean(i)) + " ")
    textfile.write("\n")

    textfile.close()

def plot(size, rmse, x_str, y_str, title):
    pyplot.plot(size,  rmse)
    print (pyplot.xlabel(x_str))
    print (pyplot.ylabel(y_str))
    print (pyplot.title(title))
    pyplot.show()

if __name__ == "__main__":  
    data = load_dataset()
    dataset_information(data)
    updated_dt, feature = one_hot_encoder(data)
    pd_data = panda_stuff(updated_dt, feature, data)
    rmse_lr = linear_regression(pd_data, data)
    rmse_bagged = bagging_regressor(pd_data, data)
    rmse_dt = decission_tree_regressor(pd_data, data)
    rmse_knn = KNeighbors_regression(pd_data, data)
    rmse_svm = SVM_reggressior(pd_data, data)
    rmse_dummy = dummy_regressor(pd_data, data)
    file_write_statistical_significance("bagged with linear: ", statistical_significance(rmse_bagged, rmse_lr))
    file_write_statistical_significance("bagged with Decision Tree: ", statistical_significance(rmse_bagged, rmse_dt))
    file_write_statistical_significance("bagged with K nearest neighbor: ", statistical_significance(rmse_bagged, rmse_knn))
    file_write_statistical_significance("bagged with SVM: ", statistical_significance(rmse_bagged, rmse_svm))
    file_write_statistical_significance("bagged with dummy: ", statistical_significance(rmse_bagged, rmse_dummy))


