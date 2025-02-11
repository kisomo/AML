

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
n = 100
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10
y2 = np.sin(x)+x/12 + np.random.randn(n)/15 #TM
y3 = np.sin(x)+x/18 + np.random.randn(n)/20  #TM
print(np.linspace(0,10,n))  #TM
print(np.random.randn(n)/5)  #TM
print(x)  #TM
import matplotlib.pyplot as plt #TM
#plt.plot(x,y, 'r--',x,y2,'bs',x,y3,'g^')  #TM
plt.plot(x,y,x,y2,x,y3,linewidth=2)  #TM
plt.xlabel("x")  #TM
plt.ylabel("y")  #TM
plt.axis([0,15,-1,3])  #TM
plt.show()  #TM

x = x.reshape(n,1) #TM

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
print(len(x)) #TM
print(len(y)) #TM
print(len(X_train)) #TM
print(len(y_test)) #TM

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    #get_ipython().magic('matplotlib notebook')
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    plt.show()
    
    
# Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
part1_scatter()


# ### Question 1
# Write a function that fits a polynomial LinearRegression model on the *training data* `X_train` for degrees 1, 3, 6, and 9.
# (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model) 
#For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. `np.linspace(0,10,100)`) 
#and store this in a numpy array. The first row of this array should correspond to the output from the model 
#trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.
# 
# The figure above shows the fitted models plotted on top of the original data (using `plot_one()`).
# *This function should return a numpy array with shape `(4, 100)`*


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    d= 100
    _output = np.zeros((4,d),dtype = np.float64) #TM
    x_input1 = np.linspace(0,10,d) #x #TM
    x_input1 = x_input1.reshape(d,1) #TM
    j = 0 #TM
    for k in [1, 3, 6, 9]: #TM
        
        poly = PolynomialFeatures(degree=k) #TM
        X_poly = poly.fit_transform(X_train) #TM
        linreg = LinearRegression().fit(X_poly, y_train) #TM
        x_input = poly.fit_transform(x_input1) #TM
        a = linreg.predict(x_input) #TM  
        _output[j,:] = a  #TM 
        j+=1  #TM
    return _output #TM


_output = answer_one()
print(_output.shape)


import matplotlib.pyplot as plt
plt.figure()
#plt.plot(x,y, 'r-',x,_output[0,:],'b-',x,_output[1,:],'g-',x,_output[2,:],'m-',x,_output[3,:],'k-')  #TM
plt.plot(x,y, alpha=0.8, lw=2, label='original') #TM
#plt.plot(X_train,y_train, alpha=0.8, lw=2, label='Training') #TM
#plt.plot(X_test,y_test, alpha=0.8, lw=2, label='Test') #TM
plt.plot(x,_output[0,:], alpha=0.8, lw=2, label='degree 1') #TM
plt.plot(x,_output[1,:], alpha=0.8, lw=2, label='degree 3') #TM
plt.plot(x,_output[2,:], alpha=0.8, lw=2, label='degree 6') #TM
plt.plot(x,_output[3,:], alpha=0.8, lw=2, label='degree 9') #TM
plt.xlabel("x")  #TM
plt.ylabel("y")  #TM
plt.title('curves')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.grid(True)
plt.axis([0,15,-1,3])  #TM
plt.legend(loc=4) #TM
plt.show()  #TM


# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()

plot_one(answer_one())


# ### Question 2
# 
# Write a function that fits a polynomial LinearRegression model on the training data `X_train` for degrees 0 through 9.
# For each model compute the $R^2$ (coefficient of determination) regression score on the training data as well as 
#the the test data, and return both of these arrays in a tuple.
# *This function should return one tuple of numpy arrays `(r2_train, r2_test)`. Both arrays should have shape `(10,)`*

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    # Your code here
    _output = np.zeros((10,2),dtype = np.float64) #TM 
    j = 0 #TM
    for k in [0,1,2,3,4,5,6,7,8,9]: #TM
        poly = PolynomialFeatures(degree=k) #TM
        X_train_poly = poly.fit_transform(X_train) #TM
        linreg = LinearRegression().fit(X_train_poly, y_train) #TM
        X_test_poly = poly.fit_transform(X_test) #TM
        _output[j,0] = linreg.score(X_train_poly, y_train)  #TM
        _output[j,1] = linreg.score(X_test_poly, y_test)  #TM
        j+=1  #TM
    return _output #TM

answer_two()
scores = answer_two()
print(pd.DataFrame(scores))


# ### Question 3
# 
# Based on the $R^2$ scores from question 2 (degree levels 0 through 9), what degree level corresponds to a model that is 
#underfitting? What degree level corresponds to a model that is overfitting? What choice of degree level would provide a 
#model with good generalization performance on this dataset? 
# 
# Hint: Try plotting the $R^2$ scores from question 2 to visualize the relationship between degree level and $R^2$. 
#Remember to comment out the import matplotlib line before submission.
# 
# *This function should return one tuple with the degree values in this order: `(Underfitting, Overfitting,
# Good_Generalization)`. There might be multiple correct solutions, however, you only need to return one possible solution,
# for example, (1,2,3).* 

def answer_three():
    
    # Your code here
    sc1 = answer_two()
    sc = pd.DataFrame(answer_two(), columns = ['train','test'])
    a = np.min(sc1[:,0])
    A = sc.index[sc['train'] == a].tolist()
    b = np.min(sc1[:,1])
    B = sc.index[sc['test'] == b].tolist()
    c = np.max(sc1[:,1])
    c2 = np.max(sc1[:,0])
    c3 = np.max(sc1[sc1[:,1] >= c2])
    C = sc.index[sc['test'] == c3].tolist()
    return A ,B,C # 0, 9, 6 # Return your answer

print(answer_three())


import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib notebook')
plt.figure(figsize=(10,5))
t = np.linspace(0,9,10)
plt.plot(t, scores[:,0], 'o', label='training data', markersize=10)
plt.plot(t, scores[:,1], '^', label='test data', markersize=10)
plt.ylim(-0.001,1)
plt.legend(loc=4)
plt.grid(True)
plt.show()


# ### Question 4
# Training models on high degree polynomial features can result in overly complex models that overfit, so we often use 
#regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.
# For this question, train two models: a non-regularized LinearRegression model (default parameters) and a regularized 
#Lasso Regression model (with parameters `alpha=0.01`, `max_iter=10000`) both on polynomial features of degree 12. Return 
#the $R^2$ score for both the LinearRegression and Lasso model's test sets.
# *This function should return one tuple `(LinearRegression_R2_test_score, Lasso_R2_test_score)`*

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Your code here
    poly = PolynomialFeatures(degree=12) #TM
    X_poly = poly.fit_transform(x) #TM
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_poly, y, random_state = 0) #TM
    linreg = LinearRegression().fit(X_train1, y_train1) #TM
    linlasso = Lasso(alpha=5, max_iter = 10000).fit(X_train1, y_train1) #TM
    A = linreg.score(X_test1, y_test1)  #TM
    B = linlasso.score(X_test1, y_test1)  #TM

    return A,B  # Your answer here

print(answer_four())


# ## Part 2 - Classification
# 
# Here's an application of machine learning that could save your life! For this section of the assignment we will be working 
#with the [UCI Mushroom Data Set](http://archive.ics.uci.edu/ml/datasets/Mushroom?ref=datanews.io) stored in `mushrooms.csv`.
# The data will be used to train a model to predict whether or not a mushroom is poisonous. The following attributes are provided:
# 
# *Attribute Information:*
# 
# 1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s 
# 2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s 
# 3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y 
# 4. bruises?: bruises=t, no=f 
# 5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s 
# 6. gill-attachment: attached=a, descending=d, free=f, notched=n 
# 7. gill-spacing: close=c, crowded=w, distant=d 
# 8. gill-size: broad=b, narrow=n 
# 9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y 
# 10. stalk-shape: enlarging=e, tapering=t 
# 11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=? 
# 12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 16. veil-type: partial=p, universal=u 
# 17. veil-color: brown=n, orange=o, white=w, yellow=y 
# 18. ring-number: none=n, one=o, two=t 
# 19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z 
# 20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y 
# 21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y 
# 22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
# 
# <br>
# 
# The data in the mushrooms dataset is currently encoded with strings. These values will need to be encoded to numeric to work 
#with sklearn. We'll use pd.get_dummies to convert the categorical variables into indicator variables. 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')

print(mush_df.shape) #TM
print(mush_df.head()) #TM

mush_df2 = pd.get_dummies(mush_df)

print("+++++++++++++++++") #TM
print(mush_df2.shape) #TM
print(mush_df2.head()) #TM

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


# ### Question 5
# 
# Using `X_train2` and `y_train2` from the preceeding cell, train a DecisionTreeClassifier with default parameters and 
#random_state=0. What are the 5 most important features found by the decision tree? 
# As a reminder, the feature names are available in the `X_train2.columns` property, and the order of the features in 
#`X_train2.columns` matches the order of the feature importance values in the classifier's `feature_importances_` property. 
# *This function should return a list of length 5 containing the feature names in descending order of importance.*
# *Note: remember that you also need to set random_state in the DecisionTreeClassifier.*

def answer_five():
    from sklearn.tree import DecisionTreeClassifier
    from adspy_shared_utilities import plot_decision_tree #TM
    from adspy_shared_utilities import plot_feature_importances #TM
    from sklearn.svm import SVC #TM

    # Your code here
    clf_dt = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 10, random_state = 0).fit(X_train2, y_train2)
    clf_SVC = SVC(kernel = 'linear', C=10).fit(X_train2, y_train2)
    #plot_decision_tree(clf, X_train2.columns, y_train2.columns)
    print('Mushroom dataset: decision tree')
    print('Accuracy of DT classifier on training set: {:.2f}' .format(clf_dt.score(X_train2, y_train2)))
    print('Accuracy of DT classifier on test set: {:.2f}'.format(clf_dt.score(X_test2, y_test2)))
    
    print('\n')

    print('Mushroom dataset: SVC')
    print('Accuracy of SVC classifier on training set: {:.2f}' .format(clf_SVC.score(X_train2, y_train2)))
    print('Accuracy of SVC classifier on test set: {:.2f}'.format(clf_SVC.score(X_test2, y_test2)))

    plt.figure(figsize=(10,6),dpi=80)
    plot_feature_importances(clf_dt, X_train2.columns )
    plt.tight_layout()

    plt.show()


    return True# Your answer here

print(answer_five())


# ### Question 6
# 
# For this question, we're going to use the `validation_curve` function in `sklearn.model_selection` to determine training and 
#test scores for a Support Vector Classifier (`SVC`) with varying parameter values.  Recall that the validation_curve function, 
#in addition to taking an initialized unfitted classifier object, takes a dataset as input and does its own internal train-test
# splits to compute results.
# **Because creating a validation curve requires fitting multiple models, for performance reasons this question will use just 
#a subset of the original mushroom dataset: please use the variables X_subset and y_subset as input to the validation curve 
#function (instead of X_mush and y_mush) to reduce computation time.**
# The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel. 
# So your first step is to create an `SVC` object with default parameters (i.e. `kernel='rbf', C=1`) and `random_state=0`.
# Recall that the kernel width of the RBF kernel is controlled using the `gamma` parameter.  
# With this classifier, and the dataset in X_subset, y_subset, explore the effect of `gamma` on classifier accuracy by 
#using the `validation_curve` function to find the training and test scores for 6 values of `gamma` from `0.0001` to `10` 
#(i.e. `np.logspace(-4,1,6)`). Recall that you can specify what scoring metric you want validation_curve to use by setting the
# "scoring" parameter.  In this case, we want to use "accuracy" as the scoring metric.
# 
# For each level of `gamma`, `validation_curve` will fit 3 models on different subsets of the data, returning 
#two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.
# 
# Find the mean score across the three models for each level of `gamma` for both arrays, creating two arrays 
#of length 6, and return a tuple with the two arrays.
# 
# e.g.
# 
# if one of your array of scores is
# 
#     array([[ 0.5,  0.4,  0.6],
#            [ 0.7,  0.8,  0.7],
#            [ 0.9,  0.8,  0.8],
#            [ 0.8,  0.7,  0.8],
#            [ 0.7,  0.6,  0.6],
#            [ 0.4,  0.6,  0.5]])
#        
# it should then become
# 
#     array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])
# 
# *This function should return one tuple of numpy arrays `(training_scores, test_scores)` where each array in the 
#tuple has shape `(6,)`.*


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # Your code here
    param_range = np.logspace(-4, 1, 6)
    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset, param_name='gamma', param_range=param_range, cv=3, scoring = "accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    return train_scores_mean, train_scores_std , test_scores_mean, test_scores_std # Your answer here

print(answer_six())



# ### Question 7  
# Based on the scores from question 6, what gamma value corresponds to a model that is underfitting 
#(and has the worst test set accuracy)? What gamma value corresponds to a model that is overfitting 
#(and has the worst test set accuracy)? What choice of gamma would be the best choice for a model with good generalization 
#performance on this dataset (high accuracy on both training and test set)? 
# 
# Hint: Try plotting the scores from question 6 to visualize the relationship between gamma and accuracy. Remember to 
#comment out the import matplotlib line before submission.
# 
# *This function should return one tuple with the degree values in this order: `(Underfitting, Overfitting, 
#Good_Generalization)` Please note there is only one correct solution.*


def answer_seven():
    
    # Your code here
    x = np.logspace(-4, 1, 6)
    y1, d1, y2, d2 = answer_six() 

    plt.title('Validation Curve with SVM')
    plt.xlabel('$\gamma$ (gamma)')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    lw = 2

    plt.semilogx(x, y1, label='Training score', color='darkorange', lw=lw)

    plt.fill_between(x, y1 - d1, y1 + d1, alpha=0.2, color='darkorange', lw=lw)

    plt.semilogx(x, y2, label='cross validation score', color='navy', lw=lw)

    plt.fill_between(x, y2 - d2, y2 + d2, alpha=0.2, color='navy', lw=lw)

    plt.legend(loc='best')
    plt.show()


    y1 = y1.reshape(len(y1),1)
    y2 = y2.reshape(len(y2),1)
    out1 = np.concatenate((y1,y2), axis =1)
    #print(out1)
    sc = pd.DataFrame(out1, columns = ['train','test'])
    #print(sc)
    x2 = pd.DataFrame(x)
    #print(x2)
    a = np.min(sc['train'])
    #print(a)
    b = np.min(sc['test'])
    #print(b)
    c = np.max(sc['test'])
    #print(c)
    A = sc.index[sc['train'] == a].tolist()
    B = sc.index[sc['test'] == b].tolist()
    C = sc.index[sc['test'] == c].tolist()
    #print(A[0],B[0],C[0])
    D1 = x2.iloc[int(A[0])]
    #print(D1)
    D2 = x2.iloc[int(B[0])]
    #print(D2)
    D3 = x2.iloc[int(C[0])]
    #print(D3)


    return float(D1),float(D2), float(D3) #0.0001, 1, 0.1 # Return your answer

#answer_seven()
print(answer_seven())






