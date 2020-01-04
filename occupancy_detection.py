#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from mlxtend.classifier import StackingClassifier

from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, roc_auc_score
from sklearn.metrics import roc_curve, average_precision_score, matthews_corrcoef, make_scorer
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif

############# START OF QUESTION 1 #############
print("Start of question 1\n\n")

data = pd.read_csv("data.csv")

y = data['Occupancy']
X = data.loc[:, data.columns != 'Occupancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

print("\n\nEnd of question 1\n\n")

############# START OF QUESTION 2 #############
print("Start of question 2\n\n")

X_train_two, X_validation, y_train_two, y_validation = train_test_split(X_train, y_train, test_size=0.20, 
                                                                        stratify=y_train,random_state=42)

print("\n\nEnd of question 2\n\n")

############# START OF QUESTION 3 #############
print("Start of question 3\n\n")

scaler = MinMaxScaler()

scaler.fit(X_train_two)
normalized_x_train_two = scaler.transform(X_train_two)
normalized_x_validation = scaler.transform(X_validation)

np.savetxt("normalized_x_train_two.csv", normalized_x_train_two, delimiter=",")
np.savetxt("normalized_x_validation.csv", normalized_x_validation, delimiter=",")

scaler.fit(normalized_x_train_two)
normalized_x_train = scaler.transform(X_train)
normalized_x_test = scaler.transform(X_test)
normalized_x_validation_with_orig = scaler.transform(X_validation)

print("\n\nEnd of question 3\n\n")

############# START OF QUESTION 4 #############
print("Start of question 4\n\n")

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_two, y_train_two)

unnormalized_accuracy = clf.score(X_validation, y_validation)

clf.fit(normalized_x_train_two, y_train_two)
normalized_accuracy = clf.score(normalized_x_validation, y_validation)

if unnormalized_accuracy > normalized_accuracy or unnormalized_accuracy == normalized_accuracy:
    print("No, I did not. Unnormalized Accuracy: {}, Normalized Accuracy: {}"
          .format(unnormalized_accuracy,normalized_accuracy))
else:
    print("Yes, I did. Unnormalized Accuracy: {}, Normalized Accuracy: {}"
          .format(unnormalized_accuracy,normalized_accuracy))

print("\n\nEnd of question 4\n\n")

############# START OF QUESTION 5 #############
print("Start of question 5\n\n")

models = [
    ("Logistic Regression",LogisticRegression(random_state=42)),
    ("K-Nearest Neighbour",KNeighborsClassifier(n_neighbors=1)),
    ("Naive Bayes",GaussianNB()),
    ("Decision Tree",DecisionTreeClassifier(random_state=42)),
    ("Random Forest",RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Support Vector Machine",SVC(kernel="rbf", C=1,gamma=0.125, random_state=42)),
    ("AdaBoostClassifier",AdaBoostClassifier(n_estimators=10, random_state=42))
]

metrics = [
    ("Accuracy Score", accuracy_score),
    ("F-Measure", f1_score),
    ("Sensitivity", recall_score),
    ("Specificity", recall_score),
    ("Precision", precision_score),
    ("Area Under ROC Curve", roc_auc_score),
    ("Area Under Precision Recall Curve", average_precision_score),
    ("MCC", matthews_corrcoef)
]

def dump_metrics(features, label, name, model):
    accuracy = 0.0
    
    for metric_name, metric in metrics:
        kfold = StratifiedKFold(n_splits=10, random_state=42)
        scorer = make_scorer(metric,pos_label=0) if metric_name == "Specificity" else make_scorer(metric)

        cv_result = cross_val_score(model,features,label.values.ravel(), cv = kfold,scoring = scorer)
        
        if metric_name == "Accuracy Score":
            accuracy = cv_result

        print("{} {}: {}".format(name, metric_name, cv_result.mean()))
    print("")
    
    return accuracy.mean()

print("Unnormalized Results:\n")
for name,model in models:
    dump_metrics(X_train_two, y_train_two, name, model)

print("\nNormalized Results:\n")
for name,model in models:
    dump_metrics(normalized_x_train_two, y_train_two, name, model)

print("\n\nEnd of question 5\n\n")

############# START OF QUESTION 6 #############
print("Start of question 6\n\n")

selectors = [
    ("VarianceThreshold Selector1", VarianceThreshold(threshold=0.1)), # First selector. Threshold was optimized beforehand.
    ("SelectKBest", SelectKBest(f_classif,k=2)), # second selector. k was optimized beforehand
    ("SelectKBest", SelectKBest(chi2,k=2)) # third selector. k was optimized beforehand
]

for selector_name,selector in selectors:    
    selected_x_train_two = selector.fit_transform(X_train_two, y=y_train_two)
    selected_normalized_x_train_two  = scaler.fit_transform(selected_x_train_two)
    
    print("{} unnormalized results:\n".format(selector_name))
    for name, model in models:
        dump_metrics(selected_x_train_two, y_train_two, name, model)

    print("{} normalized results:\n".format(selector_name))
    for name, model in models:
        dump_metrics(selected_normalized_x_train_two, y_train_two, name, model)

print("While it is possible get improvement for some models such as Naive Bayes, it is better to use all of the features for the most accurate model which is K-Nearest Neighbour.")

print("\n\nEnd of question 6\n\n")

############# START OF QUESTION 7 #############
print("Start of question 7\n\n")

name, selector = selectors[0]

optimum_x_train_two = selector.fit_transform(X_train_two, y=y_train_two)
optimum_normalized_x_train_two  = scaler.fit_transform(optimum_x_train_two)

hyper_parameters = {
    "K-Nearest Neighbour": [1] + [i*5 for i in range(1,21)],
    "Random Forest": [5,10,25,50,75,100,150,200,250,300,350,400,450,500],
    "AdaBoostClassifier": [5,10,15,20,25,30,40,50,75,100,125,150,175,200],
    "Support Vector Machine": ([2**i for i in range(-5,16,2)], [2**i for i in range(-15,6,2)])
}

opt_params = dict()

for model_name, parameters in hyper_parameters.items():
    model = None
    optimum_accuracy = 0.0
    optimum_parameters = ""
    
    if model_name == "K-Nearest Neighbour":
        for parameter in parameters:
            print("n_neighbors: {}".format(parameter))
            model = KNeighborsClassifier(n_neighbors=parameter)
            accuracy = dump_metrics(optimum_normalized_x_train_two, y_train_two, model_name, model)
            
            if accuracy > optimum_accuracy:
                optimum_accuracy = accuracy
                optimum_parameters = "n_neighbors=" + str(parameter)
                opt_params["n_neighbors"] = parameter
    elif model_name == "Random Forest":
        for parameter in parameters:
            print("n_estimators: {}".format(parameter))
            model = RandomForestClassifier(n_estimators=parameter)
            accuracy = dump_metrics(optimum_normalized_x_train_two, y_train_two, model_name, model)
            
            if accuracy > optimum_accuracy:
                optimum_accuracy = accuracy
                optimum_parameters = "n_estimators=" + str(parameter)
                opt_params["rf_estimators"] = parameter
    elif model_name == "Support Vector Machine":
        C_values = parameters[0]
        gamma_values = parameters[1]
        
        for C in C_values:
            for gamma in gamma_values:
                print("C: {}, gamma: {}".format(C, gamma))
                model = SVC(kernel="rbf", C=C,gamma=gamma)
                accuracy = dump_metrics(optimum_normalized_x_train_two, y_train_two, model_name, model)
                
                if accuracy > optimum_accuracy:
                    optimum_accuracy = accuracy
                    optimum_parameters = "C:" + str(C) + ", " + "gamma:" + str(gamma)
                    opt_params["svm_c"] = C
                    opt_params["svm_gamma"] = gamma
    elif model_name == "AdaBoostClassifier":
        for parameter in parameters:
            print("n_estimators: {}".format(parameter))
            model = AdaBoostClassifier(n_estimators=parameter)
            accuracy = dump_metrics(optimum_normalized_x_train_two, y_train_two, model_name, model)
            
            if accuracy > optimum_accuracy:
                optimum_accuracy = accuracy
                optimum_parameters = "n_estimators=" + str(parameter)
                opt_params["ab_estimators"] = parameter
    
    print("Optimum Accuracy for {}: {}".format(model_name, optimum_accuracy))
    print("Optimum Parameters:")
    print(optimum_parameters + "\n\n")

classifiers = [
    ("K-Nearest Neighbour",KNeighborsClassifier(n_neighbors=opt_params["n_neighbors"])),
    ("Random Forest",RandomForestClassifier(n_estimators=opt_params["rf_estimators"], random_state=42)),
    ("Support Vector Machine",SVC(kernel="rbf", C=opt_params["svm_c"],gamma=opt_params["svm_gamma"], random_state=42)),
    ("AdaBoostClassifier",AdaBoostClassifier(n_estimators=opt_params["ab_estimators"], random_state=42))
]

for name,model in classifiers:
    model.fit(optimum_normalized_x_train_two, y_train_two)
    preds = model.predict(X_validation)

    print("Accuracy of {} with optimal hyperparameters on validation set: {}".format(name, accuracy_score(y_validation, preds)))
    print("F-Score of {} with optimal hyperparameters on validation set: {}".format(name, f1_score(y_validation, preds)))
    print("Sensitivity of {} with optimal hyperparameters on validation set: {}".format(name, recall_score(y_validation, preds)))
    print("Specificity of {} with optimal hyperparameters on validation set: {}".format(name, recall_score(y_validation, preds,pos_label=0)))
    print("Precision of {} with optimal hyperparameters on validation set: {}".format(name, precision_score(y_validation, preds)))
    print("ROC-Auc Score of {} with optimal hyperparameters on validation set: {}".format(name, roc_auc_score(y_validation, preds)))
    print("MCC of {} with optimal hyperparameters on validation set: {}".format(name, matthews_corrcoef(y_validation, preds)))

print("\n\nEnd of question 7\n\n")

# ### 8. Implement a stacking ensemble, which combines the best performing classifiers 
# obtained in step 7 by a meta-learner (which can be logistic regression). 
# Here you will use the optimum hyper-parameters you found in step 7 to train the models 
# you selected in stacking. You can try different combinations of classifiers for this purpose. 
# Perform cross-validation and report the same accuracy measures as in step 5. 
# Then train the model on the train set 2 and test on validation set. 
# Report the accuracy measures on validation data.

lr = LogisticRegression(random_state=42)

print("Scores with optimum data:\n")

sclf = StackingClassifier(classifiers=classifiers, meta_classifier=lr)
dump_metrics(optimum_normalized_x_train_two, y_train_two, "Stacking Ensemble", sclf)

sclf.fit(X_train_two, y_train_two)

preds = sclf.predict(X_validation)

print("Scores with validation data:\n")

print("Stacking Ensemble Accuracy score: {}".format(accuracy_score(y_validation, preds)))
print("Stacking Ensemble F-Measure: {}".format(f1_score(y_validation, preds)))
print("Stacking Ensemble Sensitivity: {}".format(recall_score(y_validation, preds)))
print("Stacking Ensemble Specificity: {}".format(recall_score(y_validation, preds,pos_label=0)))
print("Stacking Ensemble Precision: {}".format(precision_score(y_validation, preds)))
print("Stacking Ensemble Area Under ROC Curve: {}".format(roc_auc_score(y_validation, preds)))
print("Stacking Ensemble MCC: {}".format(matthews_corrcoef(y_validation, preds)))


# ### 9. Generate ROC curves for the methods compared and combine these in a single plot. 
# Comment on the accuracy results. Which methods give the best performance? 
# Can you suggest other methods to further improve the accuracy?

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

for cls in classifiers:
    model = cls.fit(X_train_two, y_train_two)
    yproba = model.predict_proba(X_validation)[::,1]
    
    fpr, tpr, _ = roc_curve(y_validation,  yproba)
    auc = roc_auc_score(y_validation, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


# ### 10. Train the method that gives the most accurate predictions so far 
# (i.e. the highest overall accuracy) on the original train set after applying the best 
# feature selection and normalization strategy and compute predictions on the samples
# of the test set(s) for which the true labels are available. Report the same accuracy 
# measures as in step 5.

abc = AdaBoostClassifier(n_estimators=200, random_state=42)
dump_metrics(normalized_x_train, y_train, "Ada Boost", abc)


# ### 11. Do literature review and find publications on the same topic. 
# Which methods performed the best? Compare them with the methods you developed in this project.
# Can you improve your methods using the techniques implemented in the literature? 
# Suggest ideas for improvement.

