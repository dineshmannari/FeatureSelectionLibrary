#!/usr/bin/env python
# coding: utf-8

# In[26]:


#pip install --upgrade scikit-learn


# ## Importing libraries

# In[89]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ## Reading the dataset

# In[90]:


df = pd.read_csv('cleaned_arythmia.csv')
df


# In[91]:


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[92]:


X


# In[93]:


print(y.unique())


# In[94]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=237)


# In[95]:


X_train


# In[96]:


X_test


# In[97]:


y_train


# In[98]:


y_test


# ### Using Recursive Feature Elimination (RFE) with SelectKBeest and applying SVM

# In[99]:


classifier_names = [
    #'KNN',
    'Linear SVM',
    #'RBF SVM',
    #'Gaussian Process',
    'Decision Tree',
    'Random Forest',
    #'Neural Net',
    'AdaBoost',
    #'Naive Bayes',
    #'QDA'
]


## In[100]:


classifiers = [
    #KNeighborsClassifier(16),
    SVC(kernel="linear", C=0.025), 
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #MLPClassifier(alpha=1, max_iter = 1000),
    AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis(),
]
    


# In[83]:


accuracy = dict()
for name in classifier_names:
    accuracy[name] = list()


# In[84]:


accuracy


# In[85]:
#For Univariate Feature Selection with SelectKBest --> selector = SelectKBest(mutual_info_regression, k = K)
#For Recursive Feature Elimination --> selector = RFE(estimator=classifier,n_features_to_select = K, step = 1)
#For SelectFromModel --> selector = SelectFromModel(estimator=classifier, max_features = K)
number_of_features = int(X.shape[1]*0.1)
print(f"number_of_features = {number_of_features}")

for name, classifier in zip(classifier_names, classifiers):
    print(f"Currently on {classifier}")
    selector = selector = SelectFromModel(estimator=classifier, max_features = number_of_features)
    selector.fit(X, y)
    col = X.columns[selector.get_support()].tolist()
    temp_X_train, temp_X_test = X_train[col], X_test[col]
    classifier.fit(temp_X_train, y_train)
    y_pred = classifier.predict(temp_X_test)
    accuracy[name].append(accuracy_score(y_test, y_pred))


# In[80]:


new_df = pd.DataFrame.from_dict(accuracy)
new_df.to_csv('arrhythmia_sklearn_accuracy_sfm.csv')


# In[ ]:




