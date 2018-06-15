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

def iou(boxA, boxB):
    iou = 1 - spatial.distance.cosine(np.asarray(boxA), np.asarray(boxB))
    return iou

img_id2idx = cPickle.load(open(os.path.join('data/train36_imgid2idx.pkl')))
idx2imgid = {}
for k,v in img_id2idx.items():
    idx2imgid[v]=k
del img_id2idx
#change the name file
h5_path = os.path.join('data/train36_toy.hdf5')
with h5py.File(h5_path, 'r') as hf:
    bb = np.array(hf.get('image_features'))
    cos_distances = []
    for i in range(0,bb.shape[0]):
        name = "COCO_train2014_"  
        bb_windows = []
        cos_distance = np.zeros((36,36))
        
        for j in range(0,36):
            curr_bb = map(int,np.ceil(bb[i][j]).tolist())
            bb_windows.append(curr_bb)
        
        for m,n in itertools.combinations(range(len(bb_windows)),2):
            cos_dist= iou(bb_windows[m], bb_windows[n])
            cos_distance[m][n] = cos_dist
            cos_distance[n][m] = cos_dist
        np.fill_diagonal(cos_distance, 1)
        if i%1000==0:
            print(i)
        cos_distances.append(cos_distance.reshape(1,36,36))
    cos_distances = torch.FloatTensor(np.concatenate(cos_distances,0))
    torch.save(cos_distances, open("data/cosine_train.pt",'wb'))