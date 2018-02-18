

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fraud_data.csv') #TM
print(df.shape)  #TM
print(df.isnull().values.any()) #TM

# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# *This function should return a float between 0 and 1.* 

def answer_one():
    
    # Your code here
    frauds = df[df.Class == 1] #TM
    normal = df[df.Class == 0] #TM
    a = float(len(frauds)) #TM
    b = float(len(normal)) #TM
    instances_of_fraud =  a/(a+b) #356/(21337+356) # TM
    return instances_of_fraud # Return your answer

print(answer_one()*100)

LABELS = ["Normal", "Fraud"]
count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()



# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Question 2
# 
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as 
#the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    
    # Your code here
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix

    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_majority_predicted = dummy_majority.predict(X_test)
    confusion = confusion_matrix(y_test, y_majority_predicted)

    a = accuracy_score(y_test, y_majority_predicted)
    b = precision_score(y_test, y_majority_predicted)
    c = recall_score(y_test, y_majority_predicted)
    d = f1_score(y_test, y_majority_predicted)
    e = confusion
    
    return a,b,c,d, e # Return your answer

print(answer_two())

#TN = 5344
#FP = 0
#FN = 80
#TP = 0

# ### Question 3
# 
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is 
#the accuracy, recall, and precision of this classifier?
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*
'''
def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    # Your code here
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix

    #svm = SVC(kernel='linear', C=1).fit(X_test, y_test)
    svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
    svm_predicted = svm.predict(X_test)
    confusion = confusion_matrix(y_test, svm_predicted)

    a = accuracy_score(y_test, svm_predicted)
    b = precision_score(y_test, svm_predicted)
    c = recall_score(y_test, svm_predicted)
    d = f1_score(y_test, svm_predicted)
    e = confusion
    return a, b, c, d, e # Return your answer

print(answer_three())
'''


#(0.9961283185840708, 0.96825396825396826, 0.76249999999999996, 0.85314685314685312)
#array([[5342,    2],
#       [  19,   61]]))

#TN = 5342
#FP = 2
#FN = 19
#TP = 61

# ### Question 4
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of 
#-220 on the decision function. Use X_test and y_test.
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    # Your code here
    svm = SVC(kernel = 'rbf', C = 1e9, gamma = 1e-07).fit(X_train, y_train)
    #svm = SVC(kernel = 'rbf', C = 1e9, gamma = 1e-07).fit(X_test, y_test)
    
    svm_predicted = svm.decision_function(X_test)
    svm_predicted[svm_predicted <= -220] = 0
    svm_predicted[svm_predicted > -220] = 1
    #y2 = y_test[svm_predicted >= -220]
    conf = confusion_matrix(y_test, svm_predicted)
    #y_score_list = list(zip(y_test[0:200], svm_predicted[0:200]))
    
    return  conf # Return your answer

print(answer_four())

#[[   0 5344]
# [   0   80]]

#TN = 0
#FP = 5344
#FN = 0
#TP = 80



# ### Question 5
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability 
#estimates for X_test (probability it is fraud).
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

def answer_five():
        
    # Your code here
    from sklearn.metrics import precision_recall_curve
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    
    lr = LogisticRegression() #.fit(X_train, y_train)
    y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
    y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
    #precision, recall, thresholds = precision_recall_curve(y_test, y_proba_lr)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    from sklearn.metrics import roc_curve, auc
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    

    return 0.89,0.89 # Return your answer

print(answer_five())

'''

# ### Question 6 
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for 
#scoring and the default 3-fold cross validation.
# `'penalty': ['l1', 'l2']`
# `'C':[0.01, 0.1, 1, 10, 100]`
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
# 
# 
# *This function should return a 5 by 2 numpy array with 10 floats.* 
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to 
#reshape your raw result to meet the format we are looking for.*


def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # Your code here
    from sklearn.metrics import confusion_matrix
    lr = LogisticRegression() #.fit(X_train, y_train)
    grid_values = {'C': [0.01,0.1, 1, 10, 100], 'penalty' : ['l1','l2']}
    grid_clf = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
    grid_clf.fit(X_train, y_train)
    a = grid_clf.best_params_
    b = grid_clf.best_score_
    y2 = grid_clf.predict(X_test)
    confu = confusion_matrix(y_test, y2)
    return confu # Return your answer

print(answer_six())



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

res = []
for i in [0.01,0.1, 1, 10, 100]:
    for l in ['l1','l2']:
        grid_values = {'C': i, 'penalty' : l}
        #lr = LogisticRegression(param_grid=grid_values)
        lr = LogisticRegression()
        grid_clf = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
        grid_clf.fit(X_train, y_train)
        val = cross_val_score(grid_clf, X_train, y_train, cv=3, scoring = 'recall')
        val = np.mean(val)
        res.append(val)

print(res)
    
    
# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    get_ipython().magic('matplotlib notebook')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0)
    
GridSearch_Heatmap(answer_six())

'''