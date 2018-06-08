'''
Created on Jun 2, 2018

@author: ddua
'''
import h5py
import os
import numpy as np
import cPickle
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import torch

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    if xB < xA or yB < yA:
        return 0.0
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    try:
        overlap = boxAArea + boxBArea - interArea
        if overlap==0:
            iou=1
        else:
            iou = interArea / float(overlap)
    except Exception,e:
        print(boxA)
        print(boxB)
        print(boxAArea)
        print(boxBArea)
        print(interArea)
        exit(0)
 
    # return the intersection over union value
    return iou
    
img_id2idx = cPickle.load(open(os.path.join('/home/ddua/workspace/bottom-up-attention-vqa/data/train36_imgid2idx.pkl')))
idx2imgid = {}
for k,v in img_id2idx.items():
    idx2imgid[v]=k
del img_id2idx
h5_path = os.path.join('/home/ddua/workspace/bottom-up-attention-vqa/data/train36.hdf5')
with h5py.File(h5_path, 'r') as hf:
    bb = np.array(hf.get('image_bb'))
    distances = []
    for i in range(0,bb.shape[0]):
        name = "COCO_train2014_"  
        image_idx = str(idx2imgid[i])
        image_idx = name+''.join(['0']*(12-len(image_idx)))+image_idx+'.jpg'
        image = scipy.misc.imread("/home/ddua/workspace/bottom-up-attention-vqa/data/train2014/"+image_idx)
        bb_windows = []
        distance = np.zeros((36,36))
        for j in range(0,36):
            curr_bb = map(int,np.ceil(bb[i][j]).tolist())
            bb_windows.append(curr_bb)

        for m,n in itertools.combinations(range(len(bb_windows)),2):
            dist= iou(bb_windows[m], bb_windows[n])
            distance[m][n] = dist
            distance[n][m] = dist
            if dist>1:
                print("here")
                print(dist)
        np.fill_diagonal(distance, 1)
        if i%1000==0:
            print(i)
        distances.append(distance.reshape(1,36,36))
    distances = torch.FloatTensor(np.concatenate(distances,0))
    torch.save(distances, open("/home/ddua/workspace/bottom-up-attention-vqa/data/distance_train.pt",'wb'))

def draw_bb(curr_bb):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle((curr_bb[0],curr_bb[1]), curr_bb[2],curr_bb[3], linewidth=2,fill=False,edgecolor='r',facecolor=None)
    ax.add_patch(rect)
    plt.show()

