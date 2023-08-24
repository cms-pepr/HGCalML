"""
Debugging script to be run on predictions

Input: 
    - pred_xxxxx.bin.gz
    - some title (string)
Output: 
    - plot for beta distribution 
    - plot for coordinates distribution
"""
import sys
import os
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt


INPUTFILE = sys.argv[1]
TITLE = sys.argv[2]

with gzip.open(INPUTFILE, 'rb') as f:
    data = pickle.load(f)

features, truth, predictions = data[0]

print(predictions.keys())

beta = predictions['pred_beta']
ccoords = predictions['pred_ccoords']

fig_beta, ax_beta = plt.subplots(figsize=(20,10))
ax_beta.hist(beta)
ax_beta.set_title("Beta Distribution", fontsize=20)
ax_beta.set_yscale('log')
fig_beta.suptitle(TITLE, fontsize=30)
fig_beta.savefig(os.path.join('.', 'beta.jpg'))

fig_cords, ax_cords = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
ax_cords = ax_cords.flatten()
ax_cords[0].set_title("Cluster coordinate 0", fontsize=20)
ax_cords[0].hist(ccoords[:,0])
ax_cords[1].hist(ccoords[:,1])
ax_cords[1].set_title("Cluster coordinate 1", fontsize=20)
ax_cords[2].hist(ccoords[:,2])
ax_cords[2].set_title("Cluster coordinate 2", fontsize=20)
ax_cords[3].hist(ccoords[:,3])
ax_cords[3].set_title("Cluster coordinate 3", fontsize=20)
fig_cords.tight_layout()
fig_cords.suptitle(TITLE, fontsize=30)
fig_cords.savefig(os.path.join('.', 'cords.jpg'))


