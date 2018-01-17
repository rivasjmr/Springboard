# PCA example
#
"""
Labels
Each training and test example is assigned to one of the following labels:
0 T-shirt/top  1 Trouse  2 Pullover  3 Dress   4 Coat
5 Sandal       6 Shirt   7 Sneaker   8 Bag     9 Ankle boot 

TL;DR
Each row is a separate image
Column 1 is the class label.
Remaining columns are pixel numbers (784 total).
Each value is the darkness of the pixel (1 to 255)

"""
#
import pandas as pd
#
# Training data
df_training = pd.read_csv('C:\\Users\\rivas\\OneDrive\\Documents\\JMR\\Education\\Springboard\\Projects\\Capstone1\\fashionmnisttrain.csv')

# split data table into data Features (x) and class labels (y)
x_train = df_training.iloc[:, 1:]
y_train = df_training.iloc[:, :1]

x_train.shape
y_train.shape
x_train.head()

# Standardizing
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(x_train)

# PCA in scikit-learn
from sklearn.decomposition import PCA
pca = PCA().fit(X_std)

# Plot to find Number of components
#
import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,784,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid(True)
plt.show()

"""
The above plot shows almost 96% variance by the 300 components. Therefore
keep the first 300 component.
"""
#
sklearn_pca = PCA(n_components=300)
X_new = sklearn_pca.fit_transform(X_std)
#
X_new.shape
#
#
# *******************************************************************************

plt.scatter(X_new[:,0], X_new[:,1], c=y_train)
# plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.colorbar()
plt.show() 








