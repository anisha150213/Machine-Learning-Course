import scipy
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import models, layers
import numpy as np
from sklearn.model_selection import KFold

def load_dataset():
    data1 = fetch_openml(data_id=1489)          
    data2 = fetch_openml(data_id=1462)
    return data1, data2

def one_hot_encoder(data):
    enc = OneHotEncoder(sparse=False)
    tmp=[[x] for x in data.target]  #list of one element list
    ohe_target = enc.fit_transform(tmp)
    return ohe_target
    
def no_hidden_layers(name, input, output, data, target, epoch ):   
    # cross validation
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_accuracy = [] # list to store test accuracy of each fold
    for train, test in kfolds.split(data, target) :
        # build neural network
        nn = models.Sequential()
        nn.add(layers.Dense(output, activation='softmax', input_dim=input))
        nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        # training
        nn.fit(data.iloc[train], target[train], epochs=epoch)
        # testing
        s = nn.evaluate(data.iloc[test], target[test])
        test_fold_accuracy.append(s[1])
        print("Fold",len(test_fold_accuracy),"Accuracy =",s[1])

    print("\nTesting accuracy for all folds:",test_fold_accuracy)
    print("\nAverage testing accuracy:",np.mean(test_fold_accuracy))
    file_write_cross(name, input, output, 0, 0, test_fold_accuracy, np.mean(test_fold_accuracy))
    return test_fold_accuracy

def one_hidden_layers(name, input, output, layer1, data, target, epoch ):   
    # cross validation
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_accuracy = [] # list to store test accuracy of each fold
    for train, test in kfolds.split(data, target) :
        # build neural network
        nn = models.Sequential()
        nn.add(layers.Dense(layer1, activation='relu', input_dim=input))
        nn.add(layers.Dense(output, activation="softmax"))
        nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        # training
        nn.fit(data.iloc[train], target[train], epochs=epoch)
        # testing
        s = nn.evaluate(data.iloc[test], target[test])
        test_fold_accuracy.append(s[1])
        print("Fold",len(test_fold_accuracy),"Accuracy =",s[1])
    file_write_cross( name, input, output, layer1, 0, test_fold_accuracy, np.mean(test_fold_accuracy))
    return test_fold_accuracy


def two_hidden_layers(name, input, output, layer1, layer2, data, target, epoch):   
    # cross validation
    kfolds = KFold(n_splits=10, shuffle=True, random_state=0)
    test_fold_accuracy = [] # list to store test accuracy of each fold
    for train, test in kfolds.split(data, target) :
        # build neural network
        nn = models.Sequential()
        nn.add(layers.Dense(layer1, activation='relu', input_dim=input))
        nn.add(layers.Dense(layer2, activation='relu', input_dim=layer1))
        nn.add(layers.Dense(output, activation="softmax"))
        nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        # training
        nn.fit(data.iloc[train], target[train], epochs=epoch)
        # testing
        s = nn.evaluate(data.iloc[test], target[test])
        test_fold_accuracy.append(s[1])
        print("Fold",len(test_fold_accuracy),"Accuracy =",s[1])
    file_write_cross( name, input, output, layer1, layer2, test_fold_accuracy, np.mean(test_fold_accuracy))
    return test_fold_accuracy

def file_write_cross( name, features, target, layer1, layer2, array_acc, mean_acc):
    textfile = open("result.txt", "a")
    textfile.write("\n")
    textfile.write(str(features))
    textfile.write("\t")
    textfile.write( str(target))
    textfile.write("\t")
    textfile.write( str(layer1))
    textfile.write("\t")
    textfile.write( str(layer2))
    textfile.write("\t")
    textfile.write( str(array_acc))
    textfile.write("\t")
    textfile.write( str(mean_acc))
    textfile.write("\t")
    textfile.write( str(name))
    textfile.write("\n")
    textfile.close()

def statistical_significance(acc1, acc2):
    return scipy.stats.ttest_rel(acc1, acc2)

def file_write_statistical_significance( method, value):
    textfile = open("result_significance.txt", "a")
    textfile.write(method + " ")
    textfile.write(str(value) + " \n")
    textfile.close()

if __name__ == "__main__": 
    data1, data2 = load_dataset()
    ohe_target1 = one_hot_encoder(data1)
    ohe_target2 = one_hot_encoder(data2)

    data_1489_acc_no = no_hidden_layers("1489", 5, 2, data1.data, ohe_target1, 150 ) #1489
    data_1489_acc_one_less = one_hidden_layers("1489", 5, 2, 5, data1.data, ohe_target1, 150 ) #1489 
    data_1489_acc_one_more = one_hidden_layers("1489", 5, 2, 10, data1.data, ohe_target1, 150 ) #1489
    data_1489_acc_two = two_hidden_layers("1489", 5, 2, 10, 6, data1.data, ohe_target1, 150 ) #1489

    data_1462_acc_no = no_hidden_layers("1462", 4, 2, data2.data, ohe_target2, 74 ) #1462
    data_1462_acc_one_less = one_hidden_layers("1462", 4, 2, 5, data2.data, ohe_target2, 74 ) #1462   
    data_1462_acc_one_more= one_hidden_layers("1462", 4, 2, 10, data2.data, ohe_target2, 74 ) #1462
    data_1462_acc_two = two_hidden_layers("1462", 4, 2, 10, 6, data2.data, ohe_target2, 74 ) #1462

    file_write_statistical_significance("1489 two layer With no layer", statistical_significance(data_1489_acc_two, data_1489_acc_no))   
    file_write_statistical_significance("1489 two layer With One layerWith few", statistical_significance(data_1489_acc_two, data_1489_acc_one_less))
    file_write_statistical_significance("1489 Two layer withOne layer with more", statistical_significance(data_1489_acc_two, data_1489_acc_one_more))

    file_write_statistical_significance("1462 two layer With no layer", statistical_significance(data_1462_acc_two, data_1462_acc_no))   
    file_write_statistical_significance("1462 two layer With One layerWith few", statistical_significance(data_1462_acc_two, data_1462_acc_one_less))
    file_write_statistical_significance("1462 Two layer withOne layer with more", statistical_significance(data_1462_acc_two, data_1462_acc_one_more))   