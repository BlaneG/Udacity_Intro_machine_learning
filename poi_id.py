
# coding: utf-8

# In[50]:

#!/usr/bin/python

from IPython.display import Image
import sys
import pickle
from matplotlib import pyplot as plt
import os
import pprint
import pandas as pd
import seaborn as sns
import time

get_ipython().magic(u'matplotlib inline')


sys.path.append("../tools/")


# In[51]:

os.getcwd()


# In his presentation "Machine Learning Gremlins" Ben Hammer suggests the following process model for tackling machine learning problems:

# In[52]:

Image(filename='Machine-Learning-Process.png') 


# Common issues to look at for include:
# 
# * Data leakage
#     - introducing information about your classification target which has nothing to to with the actual target
#         - grass in background of dog photos
#         - previous prostate surgery for identifying prostate cancer
#     - to avoid:  it is important to understand what the most important variables are in your model and how they are being used
# * Overfitting
#     - to avoid:  pay close attention to training error vs validation error or k-fold cross validation
#     - make sure to split the data into training and testing sets

# #Task 0:  Explore the data
# Before getting into the machine learning exercies, we first need to load and explore the Enron dataset.

# In[53]:

#navigate up one folder with "../" prefix
#we need to deserialize, extract the data structure from byte code, using the pickle module
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


# Now let's look at the type of data structure and the size of it

# In[54]:

print ("Data type: {}".format(type(enron_data)))
data_entries = len(enron_data)
print("length of Enron dictionary is {}".format(data_entries))


# So we have a dictionary of lengh 146. Let's see what's in it:

# In[55]:

#for data in enron_data:
#    print (data, enron_data[data])


# Given that the dictionary keys for enron_data are people let's print out the list of attributes for each person:

# In[56]:

pprint.pprint(enron_data["METTS MARK"])
features = len(enron_data["METTS MARK"])
print("\n Each person in the Enron dataset has {} features".format(features))


# Now let's identify hom many poi's are in the dataset?

# In[57]:

def poi_count():
    poi_count = 0
    for i in enron_data:
        #print enron_data[i]["poi"]
        if enron_data[i]["poi"] == True:
            poi_count += 1
    return "total poi count = %s" %poi_count

poi_count()  


# With 18 poi's, we have 128 non-poi's (146-18).  Now let's look at the number of NaN's in the dataset for each feature:

# In[58]:

NaNs = {}
for person in enron_data:
    for feature in enron_data[person]:
        if enron_data[person][feature] == "NaN":
            if feature in NaNs:
                NaNs[feature] += 1.
            else:
                NaNs[feature] = 1.

pprint.pprint(NaNs)



# Several features have a lot of NaNs.  Features with the highest number of NaNs are unlikely to be useful as features in our model.

# In[59]:

#here we find the percentage of NaNs for each feature to aid in feature selection
for feature in NaNs:
    NaNs[feature] = NaNs[feature]/data_entries

pprint.pprint(NaNs)
        


# #Task 1:  Select Features

# Here is a list of all the financial features available in the dataset:

# In[60]:

financial_features = ["poi",  'bonus', 'deferral_payments', 'deferred_income', 
                      'director_fees', 'exercised_stock_options', 'expenses', 
                      'loan_advances', 'long_term_incentive', 'restricted_stock', 
                      'restricted_stock_deferred', 'salary', 'total_payments', 
                      'total_stock_value']


# The cell below could be used to remove features that have more than 40% of entries that are NaN

# In[61]:

def NaN_cutoff(cut_off):
    #let's remove from the list of financial_features any features where more than 40% of the data_entries are NaN
    for feature in NaNs:
        #print feature
        if feature in financial_features:
            if NaNs[feature] > cut_off:
                financial_features.remove(feature)

    return financial_features   

financial_features = NaN_cutoff(0.5)
financial_features


# #Task 2: Remove Outliers
# ## Explore outliers in the financial features

# In[62]:

#let's put the financial features into a pandas dataframe and visualize their relationship
df = pd.DataFrame.from_dict(enron_data, orient = "index")#navigate up one folder with "../" prefix 
df_financial = df[financial_features]
sns.pairplot(df_financial, diag_kind="kde")


# In[63]:

from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt

features = ["salary", "bonus"]


def scatter(x, y):


    #Now plot the data again
    data = featureFormat(enron_data, features)

    for point in data:
        salary = point[0]
        bonus = point[1]
        plt.scatter( salary, bonus )
        plt.xlabel(x)
        plt.ylabel(y)
        
scatter(features[0], features[1])


# It looks like we have a number of outliers as all of the data is clusstered in very small areas.  Let's take one financial data pair to investigate the outliers a little further.

# Looks like we have a big outlier.  Let's see what the max salary is for the outlier

# In[64]:

def max_feature(feature):
    feature_list = []
    for person in enron_data:
        if enron_data[person][feature] != "NaN":
            feature_list.append(enron_data[person][feature])
    return max(feature_list)
    
max_feature("salary")


# Who does this outlier belong to?

# In[65]:

def person(feature):
    for person in enron_data:
        if enron_data[person][feature] == max_feature(feature):
            return person
            #break
            
person("salary")


# Let's remove this summary statistic and replot the data

# In[66]:

def remove(feature):
    for person in enron_data:
        if enron_data[person][feature] == max_feature(feature):
            enron_data.pop(person, 0 )
            break

remove("salary")
            
scatter(features[0], features[1])


# There are stil a few outliers in the data

# In[67]:

def salary_outliers():
    outliers = []
    for person in enron_data:
        if enron_data[person]["salary"] > 1000000 or enron_data[person]["bonus"]>5000000:
            if enron_data[person]["salary"] != "NaN" and enron_data[person]["bonus"] != "NaN":
                #print person, data_dict[person]["salary"], data_dict[person]["bonus"]
                outliers.append(person)
    print outliers
    
salary_outliers()


# Since the remaining outliers are all people, we'll keep them in the dataset.  But let's just check to see who all the biggest outliers are across all of the financial features:

# In[68]:

for feature in financial_features:
    print"Person: ", person(feature),"Feature:", feature


# Now let's replot the financial features with a scatter plot matrix

# In[69]:

df = pd.DataFrame.from_dict(enron_data, orient = "index")
df_financial = df[financial_features]
sns.pairplot(df_financial, diag_kind="kde")


# ## Task 3:  Create new features
# ### 3.1 Preprocessing Data

# In[70]:

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from tester import test_classifier, dump_classifier_and_data



# ###Step 1:  Re-formate data into a set of numpy arrays and split it into training and testing sets.
# When selecting features, make sure to always train your model on a subset of the data to ensure the model isn't overfit.

# In[71]:

#guided by lesson 6 regression

#first we need to format the financial features.  
#featureFormat takes a dictionary and a list of features, and returns a numpy array for each feature
#The first feature in the feature list is the poi classification.
#All NaNs are converted to 0, and data points are removed if all of the features are zero.  
data = featureFormat( enron_data, financial_features, remove_NaN =True, remove_all_zeroes = True)
#targetFeatureSplit splits the data into the target (i.e. poi classification) and features
target, features = targetFeatureSplit(data)


feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)


# ###Step 2:  Normalization

# ####Normalize Features
# Now let's normalize the financial features before applying principle component analysis for feature reduction

# In[72]:

scaler = MinMaxScaler()

feature_train_scaled = scaler.fit_transform(feature_train)
feature_test_scaled = scaler.fit_transform(feature_test)


# ###Step 3: Dimensionality Reduction
# ####Feature Selection & Dimensionality Reduction
# Feature selection and dimensionality reduction are both processes of reducing the number of variables we consider in our analysis but the go about reducing the number of variables in different ways.  While features selection involves selecting the most important features (and excluding others from the analysis), dimensionality reduction involves combining features to make new features.  Principle component analysis is a great example of diminsionality reduction.  
# 
# [Jason Brownlee](http://machinelearningmastery.com/an-introduction-to-feature-selection/) identifies three general types of feature selection:
# 
# * Filter Methods,
# * Wrapper Methods, and
# * Embedded Methods
# 
# Embeded methods learn the best features while the model is being created.  Wrapper methods compare different combinations of features.  Filter methods apply statistical techniques to score each feature.  Below, recursive feature elimination (wrapper method) and two embedded methods - Lasso regression and linear support vector machine classification - are used to select financial features. 
# 

# ####Principle Component Analysis

# By setting n_components=3 below, we are reducing the n-dimensional space (where n=len(features)) to 3 principle component dimensions.  These principle components represent the 1st, 2nd, and 3rd dimensions of max variance and the principle components are orthogonal to each other.  If this is confusing, it should because it is impossible to visualize reducing n-dimensions to 3 dimensions.  

# In[73]:

#guided by lesson 12 mini project
from sklearn.decomposition import PCA

pca = PCA(n_components = 3)
pca.fit(feature_train_scaled)
print pca.explained_variance_ratio_ 
print sum(pca.explained_variance_ratio_)


# ####Here we show how to access and plot the first two principle components for the first two features

# In[74]:

#each pca component contains a vector transformation to get from 
#the pca back to the original features
pc_1st = pca.components_[0]
pc_2nd = pca.components_[1]
pc_3rd = pca.components_[2]


#transform the original 8 features into the principle features
feature_train_scaled_pca = pca.transform(feature_train_scaled)

#use zip here to combine each list into a list of tuples
#iterate through the list of transformed features.
#ii contains the aggregated principle components, and jj contains the 
#original data points for each feature.  
for ii, jj in zip(feature_train_scaled_pca, feature_test_scaled):
    #plot the 
    plt.scatter(pc_1st[0]*ii[0], pc_1st[1]*ii[0], color="r")
    plt.scatter(pc_2nd[0]*ii[1], pc_2nd[1]*ii[1], color="y")
    plt.scatter(jj[0], jj[1], color="b")

plt.xlabel(financial_features[1+0])
plt.ylabel(financial_features[1+1])
plt.show()
    


# ####And for comparison, we plot the top two pinciple components of the 3rd and 4th features

# In[75]:

#use zip here to combine each list into a list of tuples
#iterate through the list of transformed original features
for ii, jj in zip(feature_train_scaled_pca, feature_test_scaled):
    #plot the 
    plt.scatter(pc_1st[2]*ii[0], pc_1st[3]*ii[0], color="r")
    plt.scatter(pc_2nd[2]*ii[1], pc_2nd[3]*ii[1], color="y")
    plt.scatter(jj[2], jj[3], color="b")

plt.xlabel(financial_features[1+2])
plt.ylabel(financial_features[1+3])
plt.show()


# ####Dimensionality Reduction without normalization
# Now let's see what happens when we use PCA without applying feature scaling.  

# In[76]:

pca.fit(feature_train)

pca.explained_variance_ratio_ #this contains the eigenvalues


# So the first 3 principle components explain 80%, 12% and 6% of the variance, respectively 

# In[77]:

#extract each of the three principle components each of which contain the linear 
#transformation needed to get from the back to the principle components for each 
#of the features
pc_1st = pca.components_[0]
pc_2nd = pca.components_[1]
pc_3rd = pca.components_[2]

print pc_1st, "\n", pc_2nd, "\n", pc_3rd


# In[78]:

#transform the original 8 features into the principle features
transformed_features = pca.transform(feature_train)

#use zip here to combine each list into a list of tuples
#iterate through the list of transformed original features
for ii, jj in zip(transformed_features, feature_train):
    #ii has a length of n_components and represents a vector transformation of 
    plt.scatter(pc_1st[0]*ii[0], pc_1st[1]*ii[0], color="r")
    plt.scatter(pc_2nd[0]*ii[1], pc_2nd[1]*ii[1], color="y")
    plt.scatter(jj[0], jj[1], color="b")

plt.xlabel(financial_features[1+0])
plt.ylabel(financial_features[1+1])
plt.show()    


# What happens when we remove the "linear transformation"?

# ## Task 4 & 5:  Explore and Tune the Classifier

# ####Use SVM to classify poi's without normalization or feature selection

# In[79]:

svm = LinearSVC()

svm = svm.fit(feature_train, target_train)

test_classifier(svm, enron_data, financial_features)


# ####Exploring different approaches with pipeline notation
# Now let's use the pipeline notation to explore different combinations of normalization, feature selection, and classifiers.

# In[80]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), ('classifier', LinearSVC())])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[81]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('dim_reduction', pca), 
                     ('classifier', LinearSVC())])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[82]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('dim_reduction', pca), 
                     ('classifier', LinearSVC(C=10))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[83]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('dim_reduction', pca), 
                     ('classifier', LinearSVC(C=100))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[84]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=3)), 
                     ('classifier', LinearSVC())])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[85]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(f_regression, k=3)), 
                     ('classifier', LinearSVC())])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[86]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', RFE(Lasso())), 
                     ('classifier', LinearSVC())])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[87]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(f_regression, k=2)), 
                     ('classifier', AdaBoostClassifier(n_estimators = 100))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[88]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(f_classif, k=2)), 
                     ('classifier', AdaBoostClassifier(n_estimators = 100))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[89]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(f_regression, k=2)), 
                     ('classifier', RandomForestClassifier(n_estimators = 100))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[90]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(f_classif, k=2)), 
                     ('classifier', RandomForestClassifier(n_estimators = 100))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)


print time.time()-t


# In[91]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=4)), 
                     ('classifier', AdaBoostClassifier(n_estimators = 200))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[92]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=4)), 
                     ('classifier', RandomForestClassifier(n_estimators = 1000))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[93]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=4)), 
                     ('classifier', RandomForestClassifier(n_estimators = 200))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[94]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=4)), 
                     ('classifier', RandomForestClassifier(n_estimators = 50, max_features=3))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[95]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=4)), 
                     ('classifier', RandomForestClassifier(n_estimators = 200, max_features=3))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[96]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=4)), 
                     ('classifier', RandomForestClassifier(n_estimators = 1000, max_features=3))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[97]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=4)), 
                     ('classifier', KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', 
                                                         leaf_size=30, p=2, metric='minkowski'))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)


# In[98]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=4)), 
                     ('classifier', KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', 
                                                         leaf_size=30, p=1, metric='minkowski'))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# In[104]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=3)), 
                     ('classifier', KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', 
                                                         leaf_size=30, p=1, metric='minkowski'))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

print time.time()-t


# ####Select the tuned algorithm with good accuracy, precision and recall

# In[101]:

t= time.time()
pipeline = Pipeline([('normalization', scaler), 
                     ('feature_selection', SelectKBest(chi2, k=3)), 
                     ('classifier', RandomForestClassifier(n_estimators = 50, max_features=3))])

pipeline.fit(feature_train, target_train)
prediction = pipeline.predict(feature_test)

test_classifier(pipeline, enron_data, financial_features)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(pipeline, enron_data, financial_features)
print time.time()-t


# ###Additional methods to explore include:
# 
# * using k-fold cross-validation to improve model validation

# In[ ]:



