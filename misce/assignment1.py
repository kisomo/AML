
import numpy as np
import pandas 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer 

#The object returned by `load_breast_cancer()` is a scikit-learn Bunch object, which is similar to a dictionary. 

cancer = load_breast_cancer() 

#Print the cancer keys 

print(cancer.keys()) 

#Answer: ['target_names', 'data', 'target', 'DESCR', 'feature_names'] 

print(cancer.DESCR) 

# Print the data set description 


####################### QUESTION 0 
######################################################################### 

def answer_zero(): 
    return len(cancer['feature_names']) 

print(answer_zero())

#ANSWER: 0 

####################### QUESTION 1 
######################################################################### 

def answer_one(): 

    """converts the sklearn 'cancer' bunch
    Returns:
    pandas.DataFrame: cancer data
    """ 
    data = np.c_[cancer.data, cancer.target] 
    columns = np.append(cancer.feature_names, ["target"]) 
    return pandas.DataFrame(data, columns=columns) 

frame = answer_one() 

print(frame.shape)
print(frame.tail(3))

#Answer: (569, 31) 

print(frame.describe())


####################### QUESTION 2 
######################################################################### 

def answer_two(): 
    """calculates number of malignent and benign
    Returns:
    pandas.Series: counts of each
    """ 
    cancerdf = answer_one() 
    counts = cancerdf.target.value_counts(ascending=True) 

    #value_counts is a panda_Series function that returns an object containing 
    #counts of unique values 
    #ascending means that first value is the number of zeros (malignant) and 
    #second is the number of ones (benign) 
    counts.index = "malignant benign".split() 
    #splits the two values in 'counts' into two indexed categories. 

    return counts 
output = answer_two() 

print(output.malignant)
print (output.benign)

#Answer: 212, 357 

####################### QUESTION 3 
######################################################################### 

def answer_three(): 
    """splits the data into data and labels (i.e. independent and dependent 
    variables)
    Returns: 
    (pandas.DataFrame, pandas.Series): data, labels
    """ 
    cancerdf = answer_one() 
    X = cancerdf[cancerdf.columns[:-1]] 

    # X is all of cancer data but the last two columns. 

    y = cancerdf.target 

    # target is the label for the last column 

    return X, y 
x, y = answer_three() 

print(x.shape)

#Answer: (569, 30) 

print(y.shape)

#Answer: (569,) 

####################### QUESTION 4 
######################################################################### 

#from sklearn.model_selection import train_test_split 
from sklearn.cross_validation import train_test_split

def answer_four(): 
   """splits data into training and testing sets
   Returns:
   tuple(pandas.DataFrame): x_train, y_train, x_test, y_test
   """ 
   X, y = answer_three() 
   return train_test_split(X, y, train_size=426, test_size=143, random_state=0) 

x_train, x_test, y_train, y_test = answer_four() 

####################### QUESTION 5 
######################################################################### 

from sklearn.neighbors import KNeighborsClassifier 

def answer_five(): 
    """Fits a KNN-1 model to the data
    Returns:
    sklearn.neighbors.KNeighborsClassifier: trained data
    """ 

    X_train, X_test, y_train, y_test = answer_four() 
    model = KNeighborsClassifier(n_neighbors=3) 
    model.fit(X_train, y_train) 
    return model 

knn = answer_five() 

print(knn)


####################### QUESTION 6 

######################################################################### 
#Predict the classifier of the test data using the mean value of each feature. 
# Hint: You can use `cancerdf.mean()[:-1].values.reshape(1, -1)` which gets the 
#mean value for each feature, 
# ignores the target column, and reshapes the data from 1 dimension to 2 
#(necessary for the precict 
# method of KNeighborsClassifier). 

def answer_six(): 
    """Predicts the class labels for the means of all features
    Returns:
    numpy.array: prediction (0 or 1)
    """ 
    cancerdf = answer_one() 
    means = cancerdf.mean()[:-1].values.reshape(1, -1) 

    #New shape as (1,-1). i.e, row is 1, column unknown. we get result new shape 
    #as (1, 30) 
    #print means 
    
    model = answer_five() 
    return model.predict(means) 

predict_mean = answer_six() 

print(predict_mean) 

#Answer: [ 1.] 

####################### QUESTION 7 
######################################################################### 

def answer_seven(): 
    """predicts likelihood of cancer for test set
    Returns:
    numpy.array: vector of predictions
    """ 
    X_train, X_test, y_train, y_test = answer_four() 
    knn = answer_five() 
    return knn.predict(X_test) 

predictions = answer_seven() 
print(predictions)


print("no cancer: {0}".format(len(predictions[predictions==0]))) 

print("cancer: {0}".format(len(predictions[predictions==1]))) 
# no cancer: 51 
# cancer: 92 

####################### QUESTION 8 
######################################################################### 

def answer_eight(): 
    """calculates the mean accuracy of the KNN model
    Returns:
    float: mean accuracy of the model predicting cancer
    """ 
    X_train, X_test, y_train, y_test = answer_four() 
    knn = answer_five() 
    return knn.score(X_test, y_test) 

print(answer_eight())

#Answer: 0.916083916084 

####################### PLOT RESULTS 
######################################################################### 
# Try using the plotting function below to visualize the differet predicition 
#scores between training 
# and test sets, as well as malignant and benign cells 

def accuracy_plot(): 
    import matplotlib.pyplot as plt 
    #matplotlib notebook 
    X_train, X_test, y_train, y_test = answer_four() 
    # Find the training and testing accuracies by target value (i.e. malignant, 
    #benign) 
    mal_train_X = X_train[y_train==0] 
    mal_train_y = y_train[y_train==0] 
    ben_train_X = X_train[y_train==1] 
    ben_train_y = y_train[y_train==1] 

    mal_test_X = X_test[y_test==0] 
    mal_test_y = y_test[y_test==0] 
    ben_test_X = X_test[y_test==1] 
    ben_test_y = y_test[y_test==1] 
    #return  mal_train_X,  mal_train_y, ben_train_X, ben_train_y, mal_test_X, mal_test_y, mal_test_y, ben_test_X ,  ben_test_y
    knn = answer_five() 

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y )] 


    plt.figure() 
    # Plot the scores as a bar chart 
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868', '#55a868']) 

    # directly label the score onto the bars 

    for bar in bars: 
        height = bar.get_height() 
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'. format(height, 2), ha='center', color='w', fontsize=11) 

    # remove all the ticks (both axes), and tick labels on the Y axis 

    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft= 'off', labelbottom='on') 

    # remove the frame of the chart 

    for spine in plt.gca().spines.values(): 
        spine.set_visible(False) 

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\ nTest', 'Benign\nTest'], alpha=0.8); 

plt.title('Training and Test Accuracies for Malignant and Benign Cells', 
alpha=0.8) 
plt.show() 

accuracy_plot() 




x = "115tthg"

try:
    y = float(x)
except Exception as e:
    print(e)
    
    
print("value of y is ", y)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer 
from sklearn.cross_validation import train_test_split
cancer = load_breast_cancer() 
data = np.c_[cancer.data, cancer.target] 
columns = np.append(cancer.feature_names, ["target"]) 
cancerdf = pd.DataFrame(data, columns=columns) 
X = cancerdf[cancerdf.columns[:-1]] 
y = cancerdf.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=426, test_size=143, random_state=0)


from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors=24) 
model.fit(X_train, y_train) 
model.predict(X_test) 
model.score(X_test, y_test) 


from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_test,y_test))


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
kmeans.labels_
kmeans.predict(X_test)
res = kmeans.predict(X_test)
kmeans.cluster_centers_
#print(kmeans.score(res,y_test))
plt.scatter(X_train, c=res, s=40, cmap='viridis')



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model_gnb = gnb.fit(X_train, y_train)
y_pred = model_gnb.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))


from sklearn.svm import SVR, SVC
SVC_model = SVC(kernel = 'rbf', C = 10, gamma = 10)
SVC_model = SVC_model.fit(X_train, y_train)
y_pred = SVC_model.predict(X_test)
print(SVC_model.score(X_test,y_test))


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
dt = DecisionTreeClassifier(max_depth=6).fit(X_train, y_train)
tree_predicted = dt.predict(X_test)
confusion = confusion_matrix(y_test, tree_predicted)
print(dt.score(X_test,y_test))


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)
print(lr.score(X_test,y_test))


from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linridge = Ridge(alpha=20.0)
linridge = linridge.fit(X_train_scaled, y_train)
linridge_predicted = linridge.predict(X_test)
#confusion = confusion_matrix(y_test, linridge_predicted)
print(linridge.score(X_test,y_test))


from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)
print(linlasso.score(X_test_scaled, y_test))


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg = linreg.fit(X_train, y_train)
linreg_predicted = linreg.predict(X_test)
#confusion = confusion_matrix(y_test, linreg_predicted)
print(linreg.score(X_test,y_test))



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
clf_predicted = clf.predict(X_test)
#confusion = confusion_matrix(y_test, clf_predicted)
print(clf.score(X_test,y_test))


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf = clf.fit(X_train, y_train)
clf_predicted = clf.predict(X_test)
#confusion = confusion_matrix(y_test, clf_predicted)
print(clf.score(X_test,y_test))















import scipy.stats as stats
import statsmodels.api as sm
glm_reg = sm.GLM(y_train, X_train, family=sm.families.Binomial())
glm_model = glm_reg.fit()
y_pred = glm_model.predict(X_test)
print(glm_model.score(X_test,y_test))



from sklearn.neural_network import  MLPRegressor
MLP_model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 50, 50), random_state=1)
MLP_model.fit(X_train, y_train)
y_pred = MLP_model.predict(X_test)
print(MLP_model.score(X_test,y_test))




from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
gmm.fit(X_train)
print(gmm.means_)
print(gmm.covariances_)
X, Y = np.meshgrid(np.linspace(-4, 4), np.linspace(-4,4))
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XX)
Z = Z.reshape((50,50))
plt.figure()
plt.contour(X, Y, Z)
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()

https://github.com/jakevdp/ESAC-stats-2014/blob/master/notebooks/05.3-Density-GMM.ipynb


from sklearn.mixture import GMM
from scipy import stats
clf = GMM(4, n_iter=500, random_state=3)
clf = clf.fit(X)
xpdf = np.linspace(0, 1, 569).reshape(1, -1)
density = np.exp(clf.score(xpdf))
plt.hist(X, 80, normed=True, alpha=0.5)
plt.plot(xpdf, density, '-r')
plt.xlim(-1, 1);



https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html


# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
print(X.shape)
print(y.shape)
print(X[:3,:])
print("+++++++")
print(X[len(X)-3:,:])
X = X[:, ::-1] # flip axes for better plotting
print(X.shape)
print(X[:3,:])
print("+++++++")
print(X[len(X)-3:,:])
# Plot the data with K Means Labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
#plot
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))


kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)

from sklearn.mixture import GMM
gmm = GMM(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');

probs = gmm.predict_proba(X)
print(probs[:5].round(3))













































