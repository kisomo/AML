#!/usr/bin/python3
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import scipy as sp
import seaborn as sn 
#import graphiz as gr
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
print(answer_zero()) 


def answer_one():
    data = np.c_[cancer.data, cancer.target]
    columns = np.append(cancer.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)

print(answer_one())


def answer_two():
    cancerdf = answer_one()
    counts = cancerdf.target.value_counts(ascending=True)
    counts.index = "malignant benign".split()
    return counts

print(answer_two())

def answer_three():
    cancerdf = answer_one()
    X = cancerdf[cancerdf.columns[:-1]]
    y = cancerdf.target
    return X, y

x, y = answer_three()
assert x.shape == (569, 30)
assert y.shape == (569,)

def answer_four():
    X, y = answer_three()
    return train_test_split(X, y, train_size=426, test_size=143, random_state=0)

x_train, x_test, y_train, y_test = answer_four()
assert x_train.shape == (426, 30)
assert x_test.shape == (143, 30)
assert y_train.shape == (426,)
assert y_test.shape == (143,)

from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    #X_train, X_test, y_train, y_test = answer_four()
    
    X_train, X_test, y_train, y_test = answer_four()
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    return model
    
knn = answer_five()
assert type(knn) == KNeighborsClassifier
assert knn.n_neighbors == 1


def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    # Your code here
    #cancerdf = answer_one()
    #means = cancerdf.mean()[:-1].values.reshape(1, -1)
    model = answer_five()
    return model.predict(means)
    
answer_six()

def answer_seven():
    #X_train, X_test, y_train, y_test = answer_four()
    #knn = answer_five()
    
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.predict(X_test)

predictions = answer_seven()
assert predictions.shape == (143,)
assert set(predictions) == {0.0, 1.0}

print("no cancer: {0}".format(len(predictions[predictions==0])))
print("cancer: {0}".format(len(predictions[predictions==1])))


def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    # Your code here
    #X_train, X_test, y_train, y_test = answer_four()
    #knn = answer_five()
    return knn.score(X_test, y_test)

print(answer_eight())


#%matplotlib inline

def accuracy_plot():
    import matplotlib.pyplot as plt

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

     # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2),
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()
accuracy_plot()




    
