# To Do
  - Download the dataset(s) for your project. If a train set and a test set are not already available, randomly split the dataset into a train and a test set using stratified sampling so that 80% of the samples go to train set and 20% to test set.
  - Randomly split your train set into a validation and a new train set (called train set 2) such that the validation set contains 1/5 of the samples in original train set and the train set 2 contains the remaining. Use stratified sampling to assign features to train set 2 and validation set. This should ensure that your validation set contains samples from both classes (i.e. ciliary and non-ciliary with equal proportions)
  - Normalize features in your train set 2 and validation set using min-max scaling to interval [0,1]. For this purpose you can first normalize features in your train set 2 and use the same scaling coefficients to normalize validation set. Save the normalized versions as separate files. Repeat normalizing your original train set and use the same normalization coefficients to normalize the two test sets.
  - Perform a 10-fold cross-validation experiment for the random forest classifier on normalized and unnormalized versions of train set 2. You can set the number of trees to 100. Do you get better accuracy when you perform data normalization?
  - Perform a 10-fold cross-validation experiment on train set 2 that corresponds to the best performing normalization strategy (i.e. normalized or unnormalized) for the following classifiers:
 
    Logistic regression
    k-nearest neighbor (with k=1)
    Naïve Bayes
    Decision tree
    Random forest (number of trees=100)
    SVM (RBF kernel C=1.0 gamma=0.125)
    RBF network (number of clusters = 3)
    Adaboost (number of iterations=10)

    You can use default values for other hyper-parameters of the classifiers
Report the following accuracy measures for each of these classifiers: overall
accuracy, F-measure, sensitivity, specificity, precision, area under the ROC curve, area under the precision recall curve, MCC scores. These will be cross-validation accuracies.

  - Use three feature selection methods to select feature subsets on train set 2 and compute accuracy measures in step 5 for all the classifiers. Repeat for normalized version of train set 2. Do you get improvement in accuracy when you perform feature selection or is it better to use all of the features? Which feature selection strategy gives the best accuracy?
  - Choose the version of train set 2 that contains the optimum feature set you found in step 6 and the data for the best normalization strategy. Optimize the following hyper-parameters:

    k parameter in k-NN
    number of trees in random forest
    Number of clusters in RBF network
    Number of iterations in Adaboost
    C, gamma parameter in SVM

    Try a grid of values and choose the best value(s) that maximize the overall cross-validation accuracy.

    • For k-NN you can choose 1 5 10 15 ... 100 (with increments of 5 after k=5)
    • For number of trees in random forest you can try 5 10 25 50 75 100 150 200 250 300 350 400 450 500
    • For number of clusters in RBF network you can try: 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50
    • For the number of iterations in Adaboost you can try 5 10 15 20 25 30 40 50 75 100 125 150 175 200
    • To optimize C and gamma parameters of the SVM you can consider the following parameter grid:
    C ∈ {2 %& , 2 %( , 2 %) , 2 ) , 2 ( , 2 & , ... 2 )( , 2 )& }
    γ ∈ {2 %)& , 2 %)( , ... , 2 %) , 2 ) , 2 ( , 2 & }
    
    There are a total of 11 values for the C parameter and 11 values for the gamma parameter (a total of 121 values to consider for the (C, gamma) pair).
    
    Report the best cross-validation accuracies and optimum parameter values you found.
    
    Compute predictions on the validation set using the models trained by optimum
    hyper-parameters. Report the same accuracy measures as in step 5.
    
  - Implement a stacking ensemble, which combines the best performing classifiers obtained in step 7 by a meta-learner (which can be logistic regression). Here you will use the optimum hyper-parameters you found in step 7 to train the models you selected in stacking. You can try different combinations of classifiers for this purpose. Perform cross-validation and report the same accuracy measures as in step 5. Then train the model on the train set 2 and test on validation set. Report the accuracy measures on validation data.
  - Generate ROC curves for the methods compared and combine these in a single plot. Comment on the accuracy results. Which methods give the best performance? Can you suggest other methods to further improve the accuracy?
  - Train the method that gives the most accurate predictions so far (i.e. the highest overall accuracy) on the original train set after applying the best feature selection and normalization strategy and compute predictions on the samples of the test set(s) for which the true labels are available. How many of them are correctly predicted as ciliary. How many of them are incorrectly predicted as non-ciliary.
  - Do literature review and find publications on the same topic. Which methods performed the best? Compare them with the methods you developed in this project. Can you improve your methods using the techniques implemented in the literature? Suggest ideas for improvement.
  - Present your literature review, accuracy results (cross-validation and test data accuracies) and optimum parameters found.
