from datastructures.TrainData_TrackML import TrainData_TrackML
import os
import glob
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.express as px
import plotly.graph_objects as go

foldernames = [
    # '/afs/cern.ch/user/j/jfli/public/HGCalML/scratch/data/inputs',
	'/eos/project-d/dshep/TrackML/extracted/train_1',
    '/eos/project-d/dshep/TrackML/extracted/train_2',
    '/eos/project-d/dshep/TrackML/extracted/train_3',
    '/eos/project-d/dshep/TrackML/extracted/train_4',
    '/eos/project-d/dshep/TrackML/extracted/train_5',
]

truth_paths = glob.glob(os.path.join(foldernames[0], "*truth.csv"))
hit_paths = glob.glob(os.path.join(foldernames[0], "*hits.csv"))
cells = glob.glob(os.path.join(foldernames[0], "*cells.csv"))
particle_paths = glob.glob(os.path.join(foldernames[0], "*particles.csv"))

n_csvs = len(particle_paths)

# truth_csv = pd.read_csv(truths[0])
# hit_csv = pd.read_csv(hits[0])
# cell_csv = pd.read_csv(cells[0])
# df_particles = pd.read_csv(particle_paths[0])

n_events = 5
n_tracks = 100
matrix = pd.DataFrame()
all_pt = [] # np.array([])
chosen_csvs = []

# fig = plt.figure()
# ax = plt.axes(projection='3d')
fig = go.Figure()

for i in range(n_events):
	idx_choice = random.randint(0, n_csvs-1)
	df_particles = pd.read_csv(particle_paths[idx_choice])
	df_hits = pd.read_csv(hit_paths[idx_choice])
	df_truth = pd.read_csv(truth_paths[idx_choice])

	p = np.sqrt(df_particles['px'] ** 2 + df_particles['py'] ** 2 + df_particles['pz'] ** 2)
	ptx = np.sqrt(df_particles['px'] ** 2 + df_particles['py'] ** 2)
	
	df_particles = df_particles[(p > 8)]
	df_truth = df_truth[np.isin(df_truth['particle_id'], df_particles['particle_id'])]
	df_hits = df_hits[np.isin(df_hits['hit_id'], df_truth['hit_id'])]

	# print(len(df_particles))
	# print(len(df_hits))
	fig.add_scatter3d(x=df_hits['x'], y=df_hits['y'], z=df_hits['z'])

	# fig.add_scatter3d(connectgaps=False, go.Scatter3d(x=df_hits['x'], y=df_hits['y'], z=df_hits['z']))
	# ax.scatter3D(df_hits['x'], df_hits['y'], df_hits['z'])

	tracks = random.sample(list(p), n_tracks)
	# all_pt = np.concatenate(all_pt, tracks)
	all_pt += tracks
	# matrix[str(idx_choice)] = tracks
	chosen_csvs.append(particle_paths[idx_choice].split("/")[-1])
	# matrix.append(tracks)


print(df_hits.keys())
fig_path = '/eos/user/j/jfli/predictions/visualizations/' # '/afs/cern.ch/user/j/jfli/public/HGCalML/scratch/'
# plt.hist(all_pt)
# plt.xlabel("pT")
# plt.yscale('log')
# plt.show()
# plt.savefig(fig_path + 'hist.png')
# plt.clf()


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(df_hits['x'], df_hits['y'], df_hits['z'], 'green')
# plt.savefig(fig_path + '3d_pred.png)

fig.write_html(fig_path+"3d_pred_plotly.html")
