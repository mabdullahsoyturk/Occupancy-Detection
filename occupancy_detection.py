#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import roc_curve, average_precision_score, matthews_corrcoef, make_scorer

from sklearn.feature_selection import SelectKBest,chi2, f_classif, mutual_info_classif, VarianceThreshold


# ### 1. Download the dataset(s) for your project. If a train set and a test set are not already available, randomly split the dataset into a train and a test set using stratified sampling so that 80% of the samples go to train set and 20% to test set.
# 
# Data was retrieved from: https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
# 
# ### 2. Randomly split your train set into a validation and a new train set (called train set 2) such that the validation set contains 1/5 of the samples in original train set and the train set 2 contains the remaining. Use stratified sampling to assign features to train set 2 and validation set. This should ensure that your validation set contains samples from both classes (i.e. ciliary and non-ciliary with equal proportions)

# In[2]:


train_data = pd.read_csv("datatraining.csv")

y_train = train_data['Occupancy']
X_train = train_data.loc[:, train_data.columns != 'Occupancy']

X_train_two, X_validation, y_train_two, y_validation = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train, random_state=42)

test_data = pd.read_csv("datatest.csv") # My dataset has two test sets. This is test1.

y_test = test_data['Occupancy']
X_test = test_data.loc[:, test_data.columns != 'Occupancy']

test_data2 = pd.read_csv("datatest2.csv") # My dataset has two test sets. This is test1.

y_test2 = test_data2['Occupancy']
X_test2 = test_data2.loc[:, test_data2.columns != 'Occupancy']


# ### 3. Normalize features in your train set 2 and validation set using min-max scaling to interval [0,1]. For this purpose you can first normalize features in your train set 2 and use the same scaling coefficients to normalize validation set. Save the normalized versions as separate files. Repeat normalizing your original train set and use the same normalization coefficients to normalize the two test sets.

# In[3]:


scaler = MinMaxScaler()

scaler.fit(X_train_two)
normalized_x_train_two = scaler.transform(X_train_two)
normalized_x_validation_train_two = scaler.transform(X_validation)

np.savetxt("normalized_x_train_two.csv", normalized_x_train_two, delimiter=",")
np.savetxt("normalized_x_validation.csv", normalized_x_validation_train_two, delimiter=",")

scaler.fit(X_train)
normalized_x_train = scaler.transform(X_train)
normalized_x_test = scaler.transform(X_test)
normalized_x_test2 = scaler.transform(X_test2)
normalized_x_validation_with_orig = scaler.transform(X_validation)


# ### 4. Perform a 10-fold cross-validation experiment for the random forest classifier on normalized and unnormalized versions of train set 2. You can set the number of trees to 100. Do you get better accuracy when you perform data normalization?

# In[4]:


clf = RandomForestClassifier(n_estimators=100, random_state=42)
kfold = StratifiedKFold(n_splits=10, random_state=42)

unnormalized_accuracy = cross_val_score(clf, X_train_two, y_train_two, cv=kfold).mean()
normalized_accuracy = cross_val_score(clf, normalized_x_train_two, y_train_two, cv=kfold).mean()

if unnormalized_accuracy >= normalized_accuracy:
    print("No, I did not. Unnormalized Accuracy: {}, Normalized Accuracy: {}"
          .format(unnormalized_accuracy,normalized_accuracy))
else:
    print("Yes, I did. Unnormalized Accuracy: {}, Normalized Accuracy: {}"
          .format(unnormalized_accuracy,normalized_accuracy))


# ### 5. Perform a 10-fold cross-validation experiment on train set 2 that corresponds to the best performing normalization strategy (i.e. normalized or unnormalized) for the following classifiers: 
# 
# Logistic regression
# 
# k-nearest neighbor (with k=1)
# 
# Naïve Bayes
# 
# Decision tree
# 
# Random forest (number of trees=100)
# 
# SVM (RBF kernel C=1.0 gamma=0.125)
# 
# Linear Discriminant Analysis (This model is used instead of RBF Network)
# 
# Adaboost (number of iterations=10)
# 
# You can use default values for other hyper-parameters of the classifiers
# Report the following accuracy measures for each of these classifiers: overall
# accuracy, F-measure, sensitivity, specificity, precision, area under the ROC curve,
# area under the precision recall curve, MCC scores. These will be cross-validation
# accuracies.

# In[5]:


models = [
    ("Logistic Regression",LogisticRegression(random_state=42)),
    ("K-Nearest Neighbour",KNeighborsClassifier(n_neighbors=1)),
    ("Naive Bayes",GaussianNB()),
    ("Decision Tree",DecisionTreeClassifier(random_state=42)),
    ("Random Forest",RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Support Vector Machine",SVC(kernel="rbf", C=1,gamma=0.125, random_state=42)),
    ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
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

def dump_metrics_with_cv(features, label, name, model):
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

for name,model in models:
    print("Unnormalized Results for {}:\n".format(name))
    dump_metrics_with_cv(X_train_two, y_train_two, name, model)
    
    print("\nNormalized Results for {}:\n".format(name))
    dump_metrics_with_cv(normalized_x_train_two, y_train_two, name, model)


# ### 6. Use three feature selection methods to select feature subsets on train set 2 and compute accuracy measures in step 5 for all the classifiers. Repeat for normalized version of train set 2. Do you get improvement in accuracy when you perform feature selection or is it better to use all of the features? Which feature selection strategy gives the best accuracy?

# In[6]:


for name, model in models:
    selectors = [
        ("VarianceThreshold", VarianceThreshold()),
        ("SelectKBest with f_classif 1", SelectKBest(f_classif,k=1)),
        ("SelectKBest with chi2 1", SelectKBest(chi2,k=1)),
        ("SelectKBest with f_classif 2", SelectKBest(f_classif,k=2)),
        ("SelectKBest with chi2 2", SelectKBest(chi2,k=2)),
        ("SelectKBest with f_classif 3", SelectKBest(f_classif,k=3)),
        ("SelectKBest with chi2 3", SelectKBest(chi2,k=3)),
        ("SelectKBest with f_classif 4", SelectKBest(f_classif,k=4)),
        ("SelectKBest with chi2 4", SelectKBest(chi2,k=4)),
        ("SelectKBest with f_classif 5", SelectKBest(f_classif,k=5)),
        ("SelectKBest with chi2 5", SelectKBest(chi2,k=5)),
        ("SelectKBest with f_classif 6", SelectKBest(f_classif,k=6)),
        ("SelectKBest with chi2 7", SelectKBest(chi2,k=6)),
        ("SelectKBest with f_classif 7", SelectKBest(f_classif,k=7)),
        ("SelectKBest with chi2 7", SelectKBest(chi2,k=7))
    ]
    
    for selector_name, selector in selectors:  
        selected_x_train_two = selector.fit_transform(X_train_two, y=y_train_two)
        normalized_selected_x_train_two  = selector.fit_transform(normalized_x_train_two, y=y_train_two)

        print("{} unnormalized results:\n".format(selector_name))
        dump_metrics_with_cv(selected_x_train_two, y_train_two, name, model)

        print("{} normalized results:\n".format(selector_name))
        dump_metrics_with_cv(normalized_selected_x_train_two, y_train_two, name, model)

print("Yes, I did get improvement when I performed feature selection.")
print("SelectKBest that uses chi2 with 4 features gave the best accuracy.")


# ### 7. Choose the version of train set 2 that contains the optimum feature set you found in step 6 and the data for the best normalization strategy. Optimize the following hyperparameters:
# 
# k parameter in k-NN
# number of trees in random forest
# Number of iterations in Adaboost
# C, gamma parameter in SVM
# 
# Try a grid of values and choose the best value(s) that maximize the overall cross validation accuracy.
# 
# * For k-NN you can choose 1 5 10 15 ... 100 (with increments of 5 after k=5)
# * For number of trees in random forest you can try 5 10 25 50 75 100 150 200 250 300 350 400 450 500
# * For the number of iterations in Adaboost you can try 5 10 15 20 25 30 40 50 75 100 125 150 175 200
# 
# To optimize C and gamma parameters of the SVM you can consider the
# following parameter grid:
# 
# C ∈ {2^-5, 2^-3, 2^-1, 2^1, 2^3, 2^5, ... 2^13, 2^15}
# γ ∈ {2^-15, 2^-13, ... , 2^-1, 2^1, 2^3, 2^5}
# 
# There are a total of 11 values for the C parameter and 11 values for the gamma parameter (a total of 121 values to consider for the (C, gamma) pair).
# 
# Report the best cross-validation accuracies and optimum parameter values you found.
# 
# Compute predictions on the validation set using the models trained by optimum
# hyper-parameters. Report the same accuracy measures as in step 5.

# In[7]:


best_selector = SelectKBest(chi2,k=4)
optimum_x_train_two = best_selector.fit_transform(X_train_two, y=y_train_two)

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
            accuracy = dump_metrics_with_cv(optimum_x_train_two, y_train_two, model_name, model)
            
            if accuracy > optimum_accuracy:
                optimum_accuracy = accuracy
                optimum_parameters = "n_neighbors=" + str(parameter)
                opt_params["n_neighbors"] = parameter
    elif model_name == "Random Forest":
        for parameter in parameters:
            print("n_estimators: {}".format(parameter))
            model = RandomForestClassifier(n_estimators=parameter)
            accuracy = dump_metrics_with_cv(optimum_x_train_two, y_train_two, model_name, model)
            
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
                accuracy = dump_metrics_with_cv(optimum_x_train_two, y_train_two, model_name, model)
                
                if accuracy > optimum_accuracy:
                    optimum_accuracy = accuracy
                    optimum_parameters = "C:" + str(C) + ", " + "gamma:" + str(gamma)
                    opt_params["svm_c"] = C
                    opt_params["svm_gamma"] = gamma
    elif model_name == "AdaBoostClassifier":
        for parameter in parameters:
            print("n_estimators: {}".format(parameter))
            model = AdaBoostClassifier(n_estimators=parameter)
            accuracy = dump_metrics_with_cv(optimum_x_train_two, y_train_two, model_name, model)
            
            if accuracy > optimum_accuracy:
                optimum_accuracy = accuracy
                optimum_parameters = "n_estimators=" + str(parameter)
                opt_params["ab_estimators"] = parameter
    
    print("Optimum Accuracy for {}: {}".format(model_name, optimum_accuracy))
    print("Optimum Parameters:{}\n\n".format(optimum_parameters))


# In[9]:


classifiers = [
    ("K-Nearest Neighbour",KNeighborsClassifier(n_neighbors=opt_params["n_neighbors"])),
    ("Random Forest",RandomForestClassifier(n_estimators=opt_params["rf_estimators"], random_state=42)),
    ("Support Vector Machine",SVC(kernel="rbf", C=opt_params["svm_c"],gamma=opt_params["svm_gamma"], random_state=42)),
    ("AdaBoostClassifier",AdaBoostClassifier(n_estimators=opt_params["ab_estimators"], random_state=42))
]

def dump_metrics_without_cv(features, label, validation, name, model):
    model.fit(features, label)
    preds = model.predict(validation)
    
    print("Accuracy of {} with optimal hyperparameters on validation set: {}".format(name, accuracy_score(y_validation, preds)))
    print("F-Score of {} with optimal hyperparameters on validation set: {}".format(name, f1_score(y_validation, preds)))
    print("Sensitivity of {} with optimal hyperparameters on validation set: {}".format(name, recall_score(y_validation, preds)))
    print("Specificity of {} with optimal hyperparameters on validation set: {}".format(name, recall_score(y_validation, preds,pos_label=0)))
    print("Precision of {} with optimal hyperparameters on validation set: {}".format(name, precision_score(y_validation, preds)))
    print("ROC-Auc Score of {} with optimal hyperparameters on validation set: {}".format(name, roc_auc_score(y_validation, preds)))
    print("MCC of {} with optimal hyperparameters on validation set: {}\n\n".format(name, matthews_corrcoef(y_validation, preds)))

optimum_x_validation = best_selector.fit_transform(X_validation, y=y_validation)
    
for name,model in classifiers:
    dump_metrics_without_cv(optimum_x_train_two, y_train_two, optimum_x_validation, name, model)


# ### 8. Implement a stacking ensemble, which combines the best performing classifiers obtained in step 7 by a meta-learner (which can be logistic regression). Here you will use the optimum hyper-parameters you found in step 7 to train the models you selected in stacking. You can try different combinations of classifiers for this purpose. Perform cross-validation and report the same accuracy measures as in step 5. Then train the model on the train set 2 and test on validation set. Report the accuracy measures on validation data.

# In[10]:


classifiers_for_stacking = [
    KNeighborsClassifier(n_neighbors=opt_params["n_neighbors"]),
    RandomForestClassifier(n_estimators=opt_params["rf_estimators"], random_state=42),
    SVC(kernel="rbf", C=opt_params["svm_c"],gamma=opt_params["svm_gamma"], random_state=42, probability=True),
    AdaBoostClassifier(n_estimators=opt_params["ab_estimators"], random_state=42)
]

lr = LogisticRegression(random_state=42)

print("Scores with optimum data:\n")

sclf = StackingClassifier(classifiers=classifiers_for_stacking, meta_classifier=lr)
dump_metrics_with_cv(optimum_x_train_two, y_train_two, "Stacking Ensemble", sclf)

print("Scores with validation data:\n")
dump_metrics_without_cv(optimum_x_train_two, y_train_two, optimum_x_validation, "Stacking Ensemble", sclf)


# ### 9. Generate ROC curves for the methods compared and combine these in a single plot. Comment on the accuracy results. Which methods give the best performance? Can you suggest other methods to further improve the accuracy?

# In[11]:


result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

optimal_classifiers = [
    KNeighborsClassifier(n_neighbors=opt_params["n_neighbors"]),
    RandomForestClassifier(n_estimators=opt_params["rf_estimators"], random_state=42),
    SVC(kernel="rbf", C=opt_params["svm_c"],gamma=opt_params["svm_gamma"], random_state=42, probability=True),
    AdaBoostClassifier(n_estimators=opt_params["ab_estimators"], random_state=42),
    StackingClassifier(classifiers=classifiers_for_stacking, meta_classifier=lr)
]

for cls in optimal_classifiers:
    model = cls.fit(optimum_x_train_two, y_train_two)
    yproba = model.predict_proba(optimum_x_validation)[::,1]
    
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


# ### 10. Train the method that gives the most accurate predictions so far (i.e. the highest overall accuracy) on the original train set after applying the best feature selection and normalization strategy and compute predictions on the samples of the test set(s) for which the true labels are available. Report the same accuracy measures as in step 5.

# In[14]:


clf = RandomForestClassifier(n_estimators=opt_params["rf_estimators"], random_state=42)

optimum_x_train = best_selector.fit_transform(X_train, y=y_train)
optimum_x_test = best_selector.fit_transform(X_test, y=y_test)
optimum_x_test2 = best_selector.fit_transform(X_test2, y=y_test2)

clf.fit(optimum_x_train, y_train)

preds = clf.predict(optimum_x_test)

print("Accuracy of RandomForest with optimal hyperparameters on test set: {}".format(accuracy_score(y_test, preds)))
print("F-Score of RandomForest with optimal hyperparameters on test set: {}".format(f1_score(y_test, preds)))
print("Sensitivity of RandomForest with optimal hyperparameters on test set: {}".format(recall_score(y_test, preds)))
print("Specificity of RandomForest with optimal hyperparameters on test set: {}".format(recall_score(y_test, preds,pos_label=0)))
print("Precision of RandomForest with optimal hyperparameters on test set: {}".format(precision_score(y_test, preds)))
print("ROC-Auc Score of RandomForest with optimal hyperparameters on test set: {}".format(roc_auc_score(y_test, preds)))
print("MCC of RandomForest with optimal hyperparameters on test set: {}\n\n".format(matthews_corrcoef(y_test, preds)))

preds = clf.predict(optimum_x_test2)

print("Accuracy of RandomForest with optimal hyperparameters on test set2: {}".format(accuracy_score(y_test2, preds)))
print("F-Score of RandomForest with optimal hyperparameters on test set2: {}".format(f1_score(y_test2, preds)))
print("Sensitivity of RandomForest with optimal hyperparameters on test set2: {}".format(recall_score(y_test2, preds)))
print("Specificity of RandomForest with optimal hyperparameters on test set2: {}".format(recall_score(y_test2, preds,pos_label=0)))
print("Precision of RandomForest with optimal hyperparameters on test set2: {}".format(precision_score(y_test2, preds)))
print("ROC-Auc Score of RandomForest with optimal hyperparameters on test set2: {}".format(roc_auc_score(y_test2, preds)))
print("MCC of RandomForest with optimal hyperparameters on test set2: {}\n\n".format(matthews_corrcoef(y_test2, preds)))


# ### 11. Do literature review and find publications on the same topic. Which methods performed the best? Compare them with the methods you developed in this project. Can you improve your methods using the techniques implemented in the literature? Suggest ideas for improvement.

# ## Publications
# 
# 1. Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. Luis M. Candanedo, VÃ©ronique Feldheim. Energy and Buildings. Volume 112, 15 January 2016, Pages 28-39.
# 
# 2. Richardson, Ian & Thomson, Murray & Infield, David. (2008). A high-resolution domestic building occupancy model for energy demand simulations. Energy and Buildings. 40. 1560-1566. 10.1016/j.enbuild.2008.02.006. 
# 
# 3. S. Meyn, A. Surana, Y. Lin, S.M. Oggianu, S. Narayanan, T.A. Frewen, A sensor-utility-network method for estimation of occupancy in buildings, in: Decision and Control, 2009 held jointly with the 2009 28th Chinese Control Conference. CDC/CCC 2009. Proceedings of the 48th IEEE Conference on, IEEE, Shanghai, P.R. China, 2009, pp. 1494–1500.
# 
# 4. ] V.L. Erickson, Y. Lin, A. Kamthe, R. Brahme, A. Surana, A.E. Cerpa, M.D. Sohn, S. Narayanan, Energy efficient building environment control strategies using real-time occupancy measurements, in: Proceedings of the first ACM workshop on embedded sensing systems for energy-efficiency in buildings, ACM, Berkeley, California, 2009, pp. 19–24.
# 
# 5. ] C. Liao, P. Barooah, An integrated approach to occupancy modeling and estimation in commercial buildings, in: American Control Conference (ACC), IEEE, Baltimore, MD, 2010, pp. 3130–3135.
# 
# ## Methods implemented in the literature
# 
# Random Forest performed the best in the publications as well. Since this is a very well established problem, almost all methods that I have performed was already performed by other researchers. So, the results are very similiar because of that. I can definitely improve my methods using the tecniques implemented in the literature. For instance, I can use bootstrap sampling for evaluating just like in the first article. I could also optimize the parameters of linear discriminant analysis which gives a fairly good accuracy.
