from sklearn import datasets
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

def load_dataset():
    dia = datasets.fetch_openml(data_id=1050)
    return dia

def train_extra(dia):
    mytree = tree.DecisionTreeClassifier(criterion='entropy')  
    mytree.fit(dia.data, dia.target)
    print("target")
    print((dia.target_names))
    predictions = mytree.predict(dia.data)
    print(predictions)
    print(metrics.accuracy_score(dia.target,predictions))
    metrics.f1_score(dia.target, predictions, pos_label='TRUE')
    metrics.precision_score(dia.target, predictions, pos_label='TRUE')
    metrics.recall_score(dia.target, predictions, pos_label='TRUE')
    pp=mytree.predict_proba(dia.data)
    pp[:,1]
    print(metrics.roc_auc_score(dia.target, pp[:,1]))

def decision_tree_default(dia):
    dtc = tree.DecisionTreeClassifier()    
    cv = model_selection.cross_validate(dtc, dia.data, dia.target, cv = 10, scoring= 'roc_auc', return_train_score = True)
    file_write(cv, "Default DT")
    print(cv)

def decision_tree_min_samples(dia):   
    dtc = tree.DecisionTreeClassifier(min_samples_leaf= 10)
    parameters = [{"min_samples_leaf":[2,4,6,8,10]}]
    
    tuned_dtc = model_selection.GridSearchCV(dtc,parameters,scoring="roc_auc", cv= 5)
    cv = model_selection.cross_validate(tuned_dtc,dia.data, dia.target, scoring= 'roc_auc', cv=10, return_train_score = True)
    file_write(cv, "Tuned DT")
    print("Test")
    print(tuned_dtc.fit(dia.data, dia.target))
    print("best",tuned_dtc.best_params_)
    print(cv)

def random_forest(dia):
    rf = RandomForestClassifier()
    cv_rf = model_selection.cross_validate(rf, dia.data, dia.target, scoring= 'roc_auc', cv=10, return_train_score = True)
    file_write(cv_rf, "Random Forest")
    print(cv_rf)

def bagged(dia):
    bagged_dtc = BaggingClassifier()
    cv_bagged = model_selection.cross_validate(bagged_dtc, dia.data, dia.target, scoring= 'roc_auc', cv=10, return_train_score = True)
    file_write(cv_bagged, "Bagging")
    print(cv_bagged)

def adaboost(dia):
    ada_dtc = AdaBoostClassifier()
    cv_ada = model_selection.cross_validate(ada_dtc, dia.data, dia.target, scoring= 'roc_auc', cv=10, return_train_score = True)
    file_write(cv_ada, "Adaboost")
    print(cv_ada)
    

def file_write(cv, method_name):
    textfile = open("Result.txt", "a")
    textfile.write(method_name + "\n")
    
    for element in cv:   
        textfile.write(element +": " + str(cv[element]) + " \n")
 
    
    textfile.write(method_name+" fit_time: " + str(cv['fit_time'].mean()) + " \n")
    textfile.write(method_name+" score_time: " + str(cv['score_time'].mean()) + " \n")
    textfile.write(method_name+" test_score: " + str(cv['test_score'].mean()) + " \n")
    textfile.write(method_name+" train_score: " + str(cv['train_score'].mean()) + " \n")
    textfile.close()



if __name__ == "__main__":  
    data = load_dataset()
    print("\nDataset Loaded")

    print("\nTrain and check Decision tree where criterion = entropy")
    train_extra(data)

    print("\nDefault Decision Tree")
    decision_tree_default(data)

    print("\nTuned Decision Tree")
    decision_tree_min_samples(data)

    print("\nRandom Forest")
    random_forest(data)

    print("\nBagged Decision Tree")
    bagged(data)

    print("\nAdaboost Decision Tree")
    adaboost(data)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
