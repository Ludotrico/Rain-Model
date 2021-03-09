#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all required libraries
import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

np.random.seed(0)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading Data

# In[50]:


path = "C:/Users/hyunm/OneDrive/Documents/GitHub/Rain-Model/data/"
# Load the training data
X = np.genfromtxt(path + 'X_train.txt', delimiter=None)
Y = np.genfromtxt(path + 'Y_train.txt', delimiter=None)

# Test features
Xte = np.genfromtxt(path + 'X_test.txt', delimiter=None)

# Split into train and validation
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y) # Default is 80% training/20% validation
Xtr, Ytr = ml.shuffleData(Xtr, Ytr)


# ## Random Forest

# In[8]:


# computeRandomForest(num_bags, num_features):
#     Returns bags (trees) using Random Forest algorithm given num_bags and num_features
def computeRandomForest(num_bags, num_features):
    np.random.seed(0)  # Resetting the seed in case you ran other stuff.
    bags = []
    n_bags = num_bags
    num_records = Xtr.shape[0]
    
    for l in range(n_bags):
        
        # Each boosted data is the size of the original data
        Xi, Yi = ml.bootstrapData(Xtr, Ytr, num_records)


        # Train the model on that draw
        tree = ml.dtree.treeClassify(Xi, Yi, minParent=2**6, maxDepth=25, nFeatures=num_features)
        bags.append(tree)

        tr_auc = tree.auc(Xtr, Ytr)
        val_auc = tree.auc(Xva, Yva)

        print("Decision Tree : {0}".format(l))
        print("{0:>15}: {1:.4f}".format('Train AUC', tr_auc))
        print("{0:>15}: {1:.4f}".format('Validation AUC', val_auc))
        
    return bags


# ## Creating a Bagged Class

# In[10]:


class BaggedTree(ml.base.classifier):
    def __init__(self, learners):
        """Constructs a BaggedTree class with a set of learners. """
        self.learners = learners
    
    def predictSoft(self, X):
        """Predicts the probabilities with each bagged learner and average over the results. """
        n_bags = len(self.learners)
        preds = [self.learners[l].predictSoft(X) for l in range(n_bags)]
        return np.mean(preds, axis=0)


# In[40]:


# best number of features is 6
best_num_features = 6


# In[43]:


bags_100 = computeRandomForest(100, best_num_features)


# # ## Compute AUC score for given bags

# In[45]:


# AUC score for bag num 10
# print("Print AUC score for bag num: " + str(len(bags_10)))
# bt = BaggedTree(bags_10)
# bt.classes = np.unique(Y)

# print("{0:>15}: {1:.4f}".format('Train AUC', bt.auc(Xtr, Ytr)))
# print("{0:>15}: {1:.4f}".format('Validation AUC', bt.auc(Xva, Yva)))

# AUC score for bag num 30
# print("Print AUC score for bag num: " + str(len(bags_30)))
# bt2 = BaggedTree(bags_30)
# bt2.classes = np.unique(Y)

# print("{0:>15}: {1:.4f}".format('Train AUC', bt2.auc(Xtr, Ytr)))
# print("{0:>15}: {1:.4f}".format('Validation AUC', bt2.auc(Xva, Yva)))

print("Print AUC score for bag num: " + str(len(bags_100)))
bt3 = BaggedTree(bags_100)
bt3.classes = np.unique(Y)

print("{0:>15}: {1:.4f}".format('Train AUC', bt3.auc(Xtr, Ytr)))
print("{0:>15}: {1:.4f}".format('Validation AUC', bt3.auc(Xva, Yva)))


# best_num_features = 6 (computed using square root of 14 (total num features) and adding 3)
# For bags = 10
#     Train AUC: 0.8914
#     Validation AUC: 0.7702 
#     
# For bags = 30
#     Train AUC: 0.9022
#     Validation AUC: 0.7793 
#     
# For bags = 100
#     Train AUC: 0.9076
#     Validation AUC: 0.7839 (BEST)

# ## Predicting probability using test data, predict soft for 100 bags, 6 features for random root

# In[46]:


probs = bt3.predictSoft(Xte)
print(probs)


# ## Submitting Predictions

# In[47]:


# Create the data for submission by taking the P(Y=1) column from probs and just add a running index as the first column.
Y_sub = np.vstack([np.arange(Xte.shape[0]), probs[:, 1]]).T

# We specify the header (ID, Prob1) and also specify the comments as '' so the header won't be commented out with
# the # sign.
np.savetxt('Y_sub.txt', Y_sub, '%d, %.5f', header='ID,Prob1', comments='', delimiter=',')

