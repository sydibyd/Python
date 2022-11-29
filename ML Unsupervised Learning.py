# Test Unsupervised Learning algorithms
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

# Let's look at these numbers as vectors plottes on top of the data:
plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.5)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
plt.axis('equal');

# As showed in image, one vector is longer than other,
# that means the "important" of each direction.
# Knowing that the second principal component could be completely ignored
# with no much loss of information,
# may be interesting to see what the data look like by keeping 95% of the variance:
clf = PCA(0.95) # keep 95% of variance
X_trans = clf.fit_transform(X)
print(X.shape)
print(X_trans.shape)

# We are compressing the data by throwing away 5% of the variance.
# Voici the data after the compression:
X_new = clf.inverse_transform(X_trans)
plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.2)
plt.plot(X_new[:, 0], X_new[:, 1], 'ob', alpha=0.8)
plt.axis('equal');

# The dark points are the projected version.
# We see that the most important features of data are saved, and we have compressed the data by 50%.
# This is the puissance of "dimensionality reduction" : By approximating a data set in a lower dimension,
# we have an easier time visualizing it or fitting complicated models to the data.



# Application of PCA to Digits
# Let's try the application of PCA on digits data (the same data points):
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

pca = PCA(2)  # project from 64 to 2 dimensions
Xproj = pca.fit_transform(X)
print(X.shape)
print(Xproj.shape)

plt.scatter(Xproj[:, 0], Xproj[:, 1], c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar();

# It is easier to look at the relationships between the digits.
# The optimal stretch and rotation in finded in 64-dimentional space,
# that permit to see the layout of digits with no reference to  the labels.


