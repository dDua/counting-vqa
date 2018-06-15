import h5py
import numpy as np
from tempfile import TemporaryFile
h5_path = '/home/ddua/workspace/bottom-up-attention-vqa/data/val36.hdf5'

with h5py.File(h5_path, 'r') as hf:
    features = np.array(hf.get('image_features'))
feature_indices_many = np.array([18089, 30184, 26761, 21634, 31327])

feature_indices_several = np.array([38311, 19697, 1404, 38192, 28506]) 
feature_indices_few = np.array([20519, 25129, 7534, 5032, 23436]) 
feature_indices_no = np.array([14471, 29896, 10545, 20688, 22317])

many = features[feature_indices_many]
several = features[feature_indices_several]
few = features[feature_indices_few]
no = features[feature_indices_no]
features_clustering = np.concatenate((many, several, few, no), axis = 0)
print(features_clustering.shape)

# outfile = TemporaryFile()
np.save(open('features_quant','wb'), features_clustering)

np.load(open('features_quant','rb'))