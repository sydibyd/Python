# Test Unsupervised Learning algorithms.
# Editor: Saeis Sharify
# Date: 01/2018
#
#
#
# Unsupervised Learning: Dimensionality Reduction and Clustering
# Here we have a set of initial imports:
from __future__ import print_function, division
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# The goal is reducting dimentionality in data using Principal Compenent Analysis.
# Here we have a two-dimentional dataset:
np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T
plt.plot(X[:, 0], X[:, 1], 'o')
plt.axis('equal');

# As is seen in image, we have a definite trend in data. 
# Now we are goint to find the principal axes in th data with PCA:
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.components_)





