
#https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


import pickle
from scipy import stats
import seaborn as sns
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
df = pd.read_csv('fraud_data.csv')

print(df.shape)
print(df.isnull().values.any())
print(df.head(3))

count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

print(count_classes.shape)

frauds = df[df.Class == 1]
normal = df[df.Class == 0]

print(frauds.shape)
print(normal.shape)
#How different are the amount of money used in different transaction classes?
print(frauds.Amount.describe())
print(normal.Amount.describe())

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(frauds.Amount, bins = bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
#plt.xlim((0, 20000))
plt.xlim((0, 2000))
plt.yscale('log')
plt.show()

'''
#Do fraudulent transactions occur more often during certain time?
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(frauds.Time, frauds.Amount)
ax1.set_title('Fraud')

ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()
'''

from sklearn.preprocessing import StandardScaler
#df = df.drop(['Time'], axis=1)
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
data = df

#train on normal transactions
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values
print(X_train.shape)

#++++++++++++++++++++++++++++++++++++++++++++++ keras +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 5 #100
batch_size = 32

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0, save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


autoencoder = load_model('model.h5')

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df.describe()

fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)


fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df[error_df['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)



fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()

plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different threshold values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()

threshold = 2.9
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


#+++++++++++++++++++++++++++++++++++ tflearn +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import fully_connected, dropout, flatten


def my_model(data):

    layer1 = fully_connected(data, n_units = 29, activation='relu', bias = True)
    
    layer2 = fully_connected(layer1, n_units =14, activation='relu', bias = True)

    layer3 = fully_connected(layer2, n_units = 14, activation='relu', bias = True)

    layer4 = fully_connected(data, n_units = 29, activation='relu', bias = True)

    #w = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
    #b = tf.Variable(tf.constant(1.0, shape = [10]))

    #logits = tf.matmul(layer4_fccd, w) + b
    return layer4 # logits


#number of iterations and learning rate
batch_size = 100
num_steps = 20
display_step = 5
learning_rate = 0.5 #0.001

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a tensorflow friendly form. 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 29))
    #tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.constant(X_test, tf.float32)
 
    #3. The model used to calculate the logits (predicted labels)
    model = my_model
    logits = model(tf_train_dataset)

    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    #5. The optimizer is used to calculate the gradients of the loss function 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('leNet5 via API - Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
 
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)

'''
#++++++++++++++++++++++++++++++++++++++++ NAIVE BAYES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Bernoulli
BernNB = BernoulliNB(binarize = True)
BernNB.fit(X_train, y_train)
y_pred = BernNB.predict(X_test)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))

print("Bernoulli" ,acc, prec, rec, conf)

'''
#Multinomial
MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)
y_pred = MultiNB.predict(X_test)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Mutinomial", acc, prec, rec, conf)
'''


#Gaussian
GaussNB = GaussianNB()
GaussNB.fit(X_train, y_train)
y_pred = GaussNB.predict(X_test)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))

print("Gaussian", acc, prec, rec, conf)

#Bernoulli
BernNB = BernoulliNB(binarize = 0.1)
BernNB.fit(X_train, y_train)
y_pred = BernNB.predict(X_test)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))

print("Bernoulli" ,acc, prec, rec, conf)

#++++++++++++++++++++++++++++++++++++++++++++++++ K-means +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
kmeans.labels_
kmeans.predict(X_test)
y_pred = kmeans.predict(X_test)
kmeans.cluster_centers_
#print(kmeans.score(res,y_test))
#print(res)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)

print("K-means" ,acc, prec, rec, conf)

#+++++++++++++++++++++++++++++++++++++++++++++++++++ knn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

k = X_train.shape[0]
k = int(np.sqrt(k))
from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors=k) 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
#print(model.score(X_test, y_test))
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))

print("K-nn" ,acc, prec, rec, conf)

#+++++++++++++++++++++++++++++++++++++++++++ Nearest Centroid ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.predict(X_test))
print(clf.score(X_test,y_test))
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Nearest Centroid" ,acc, prec, rec, conf)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++ EM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
gmm.fit(X_train)
#print(gmm.means_)
#print(gmm.covariances_)
y_pred = gmm.predict(X_test)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
print("EM" ,acc, prec, rec, conf)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ GB Classifier +++++++++++++++++++++++++++++++++++++++++++++++++++++

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#confusion = confusion_matrix(y_test, clf_predicted)
#print(clf.score(X_test,y_test))
#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
print("GB Classifier" ,acc, prec, rec, conf)


#++++++++++++++++++++++++++++++++++++++++++++++++++++ SVC ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
from sklearn.svm import SVC
SVC_model = SVC(kernel = 'rbf', C = 10, gamma = 10)
SVC_model = SVC_model.fit(X_train, y_train)
y_pred = SVC_model.predict(X_test)
#print(SVC_model.score(X_test,y_test))
#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
conf = confusion_matrix(y_test, y_pred)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
print("SVC" ,acc, prec, rec, conf)

'''


print("+++++++++++++++++++++++++++++++= spark ++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#https://spark.apache.org/docs/2.2.0/mllib-linear-methods.html
'''
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("data/mllib/sample_svm_data.txt")
print(data.shape)

parsedData = data.map(parsePoint)

# Build the model
model = SVMWithSGD.train(parsedData, iterations=100)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "target/tmp/pythonSVMWithSGDModel")
sameModel = SVMModel.load(sc, "target/tmp/pythonSVMWithSGDModel")
'''
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://spark.apache.org/docs/2.2.0/api/python/pyspark.mllib.html#pyspark.mllib.classification.SVMWithSGD

from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkFiles

#from pyspark import SparkContext(master=None, appName=None, sparkHome=None, pyFiles=None,
 #environment=None, batchSize=0, serializer=PickleSerializer(), conf=None, gateway=None, jsc=None, 
 #profiler_cls=<class 'pyspark.profiler.BasicProfiler'>)

from pyspark import SparkContext as sc

data = [
     LabeledPoint(0.0, [0.0, 1.0]),
     LabeledPoint(1.0, [1.0, 0.0]),
     ]
lrm = LogisticRegressionWithSGD.train(sc.parallelize(data), iterations=10)
print(lrm.predict([1.0, 0.0]))
'''
#1
lrm.predict([0.0, 1.0])

lrm.predict(sc.parallelize([[1.0, 0.0], [0.0, 1.0]])).collect()
[1, 0]
lrm.clearThreshold()
lrm.predict([0.0, 1.0])

'''





