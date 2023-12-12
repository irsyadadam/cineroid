#stl
import os
import warnings

#data handling
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

#stats
import scipy
import sklearn

import numpy as np
from sklearn.model_selection import StratifiedKFold

#network
import networkx as nx

#vis
import matplotlib.pyplot as plt
import seaborn as sns

#torch

sns.set(font_scale = 1.5)
sns.set_theme()

#data preprocess
import h5py
import cv2
from PIL import Image


if __name__ == '__main__':

    #read in the datasets
    DATA_ROOT = "/data/"

    metadata_file = DATA_ROOT + "metadata.csv"
    data_file = h5py.File(DATA_ROOT + "dataset.hdf5")

    f = h5py.File(data_file, 'r')
    metadata_df = pd.read_csv(metadata_file, index_col = [0])

    #process csv
    label_df = pd.DataFrame(list(f["annot_id"]), columns = ["annot_id"])
    label_df["annot_id"] = [s.decode() for s in label_df["annot_id"].to_list()]
    
    #populate the labels column using histopath diagnosis from metadata for, using nodule-wise labels
    label_list = []
    for i in range(len(label_df["annot_id"])):
        id_ = label_df["annot_id"][i]
        label_list.append(metadata_df["histopath_diagnosis"].loc[id_])
    label_df["final_diagnosis"] = label_list

    #assign folds
    test_folds = []

    fold_num = np.zeros(len(label_df))
    
    skf = StratifiedKFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(skf.split(label_df["annot_id"].to_list(), label_df["final_diagnosis"].to_list())):
        fold_num[test_index.tolist()] = int(i)

    label_df["foldnum"] = fold_num

    #save 
    label_df.to_csv(DATA_ROOT + "label_df.csv", index = False)