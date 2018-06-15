'''
Created on Jun 2, 2018
@author: mabdrakh
'''
import h5py
import os
import numpy as np
import cPickle
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import spatial
import itertools
import torch
from scipy.cluster import  hierarchy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

features = np.load(open('features_quant','rb'))
(num_samples, num_boxes, num_features) = features.shape
threshold = 0.4
r = np.zeros(num_samples)
for i in range(0, num_samples):
    Z = hierarchy.linkage(features[i],"complete", metric="cosine")
    C = hierarchy.fcluster(Z, threshold, criterion="distance")
    unique, counts = np.unique(C, return_counts=True)
    r[i] = max(counts)/float(max(unique))
    print(r[i])
plt.plot(range(1, num_samples+1), r)
plt.xticks(range(1, num_samples+1))
plt.ylabel('R')
plt.xlabel('samples')
plt.show()
