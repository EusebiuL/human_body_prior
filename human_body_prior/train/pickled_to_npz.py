
import pickle as pkl
import numpy as np

with open('/Users/eusebiu/Downloads/mano_v1_2/models/SMPLH_male.pkl', 'rb') as f:
    data = pkl.load(f)
    np.savez(data)