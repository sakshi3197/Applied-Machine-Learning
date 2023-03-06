#!/usr/bin/env python
# coding: utf-8

# # Question 2

# In[6]:


import numpy as np
import pandas as pd


# ### 1.
# ### After reading the additional material, I would choose Multinomial Naive Bayes as my model. 

# ### Getting and merging data 

# In[9]:


data=pd.read_csv('us.txt', header=None)
data1=pd.read_csv('greek.txt', header=None)
data2=pd.read_csv('japan.txt',header=None)
data3=pd.read_csv('arabic.txt',header=None)


# In[10]:


data = data.rename(columns={0: 'Name'})


# In[11]:


data1 = data1.rename(columns={0: 'Name'})
data2 = data2.rename(columns={0: 'Name'})
data3 = data3.rename(columns={0: 'Name'})


# In[12]:


data['cat'] = 1
data1['cat'] = 2
data2['cat'] = 3
data3['cat'] = 4


# In[13]:


frames = [data, data1, data2, data3]

result = pd.concat(frames)


# In[14]:


result


# ### Spliting them into training (70%) and testing (30%) with shuffle = True

# In[15]:


from sklearn.model_selection import train_test_split
X = result.iloc[:, 0]
y = result.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, shuffle = True)


# ### Using CountVectorizer

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[17]:


XX_train_1=cv.fit_transform(X_train.values)


# In[18]:


XX_train_1= XX_train_1.toarray()


# In[19]:


XX_train_1.shape


# In[20]:


tr=[X_train, y_train]
train = pd.concat(tr, axis=1)


# In[21]:


train


# ### Implementing Multinomial Naive Bayes
# ### Getting Prior and Conditional Probabilities 

# In[22]:


def get_priors(train):
    pts = np.zeros(4)
    condi = np.zeros((4,XX_train_1.shape[1]),dtype=np.float64)
    for i in range(4):
        temp = XX_train_1[train['cat'] == (i + 1)]
        pts[i] = len(temp) / XX_train_1.shape[0]
        ##print(XX_train_1.shape)
        condi[i,:] = np.mean(temp,axis=0) + 1e-6
        
    return pts, condi


# In[23]:


from sklearn import metrics
def predict_single(X,pri,condi):
    all_probabilities= []
    for i in range(4):
        poste = np.log(condi[i]) * X
        total_poste = poste.sum()
        total_probability = total_poste + np.log(pri[i])
        all_probabilities.append(total_probability)
        
    return np.argmax(all_probabilities) + 1
        

def GNaiveBayes(X,y):
    pri,condi = get_priors(train)
    X = cv.transform(X)
    X = X.toarray()
    all_preds = [predict_single(xi,pri,condi) for xi in X]
    
    return metrics.accuracy_score(y,all_preds)


# In[24]:


GNaiveBayes(X_test, y_test)


# #### Accuracy is 0.9208333333333333 (As shown above)
# 

# References:
# [1]: Brownlee, Jason. “Metrics to Evaluate Machine Learning Algorithms in Python.” MachineLearningMastery.com, 30 Aug. 2020, https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/. 
# [2]. Team, Great Learning. “Multinomial Naive Bayes Explained.” Great Learning Blog: Free Resources What Matters to Shape Your Career!, 31 Oct. 2022, https://www.mygreatlearning.com/blog/multinomial-naive-bayes-explained/. 
# [3]: Kharwal, Aman. “Multinomial Naive Bayes in Machine Learning: Aman Kharwal.” Thecleverprogrammer, 6 Aug. 2021, https://thecleverprogrammer.com/2021/08/06/multinomial-naive-bayes-in-machine-learning/. 
# [4]. 
