import pdb
import os
import sys
import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import extra_plots as ep

# get environment variable 'HGCALML'
try:
    HGCALML = os.environ['HGCALML']
    from DeepJetCore.DataCollection import DataCollection
    from DeepJetCore.dataPipeline import TrainDataGenerator
except KeyError:
    HGCALML = None
    print("HGCALML not set, relying on gzip/pickle")


def djcdc_to_dataframe(input_path, n_events):
    """
    Converts a .djcdc to pandas dataframes for truth and features
    """

    if not os.path.exists(input_path):
        print(input_path, " not found")
        exit(1)

    dc = DataCollection(input_path)
    input_file = dc.dataDir + dc.samples[0]
    td = dc.dataclass()
    td.readFromFileBuffered(input_file)

    gen = TrainDataGenerator()
    gen.setBatchSize(1)
    gen.setSkipTooLargeBatches(False)
    gen.setSquaredElementsLimit(False)
    gen.setBuffer(td)
    n_steps = gen.getNBatches()

    truth_list = []
    feature_list = []
    generator = gen.feedNumpyData()

    for i in range(n_steps):
        if i >= n_events: break
        data = next(generator)[0]
        # print(len(data))
        # print(td.createFeatureDict(data)['recHitX'].shape)
        truth_list.append(td.createTruthDict(data))
        feature_list.append(td.createFeatureDict(data))

    features = ep.dictlist_to_dataframe(feature_list)
    truth = ep.dictlist_to_dataframe(truth_list)
    features = features.drop(columns=['event_id'])
    output_df = pd.concat([features, truth], axis=1)
    return output_df


def make_figure():
    fig = go.Figure(
        layout=go.Layout(
            width=1200,
            height=1200,
            title=go.layout.Title(
                text="3D Scatter Plot of RecHits",
                font=dict(
                    size=50,
                ),
                xref="paper",
                x=0.5,
            ),
            scene=go.layout.Scene(
                xaxis=go.layout.scene.XAxis(
                    title=go.layout.scene.xaxis.Title(text="Z")
                ),
                yaxis=go.layout.scene.YAxis(
                    title=go.layout.scene.yaxis.Title(text="Y")
                ),
                zaxis=go.layout.scene.ZAxis(
                    title=go.layout.scene.zaxis.Title(text="X")
                ),
            ),
        ),
    )
    return fig



def dataframe_to_plot(df, id=0, truth=True, clusterspace=False, verbose=False):
    df = df[df.event_id == id]
    size = 100 * np.log(df.recHitEnergy + 1)
    # change sizes bigger than 5 to 5
    size[size > 10] = 10
    size[size < 0.1] = 0.1

    if verbose:
        print(np.min(size), np.median(size), np.mean(size), np.max(size))

    fig = make_figure()

    if truth:
        ids = np.unique(df.truthHitAssignementIdx)
    else:
        ids = np.unique(df.pred_sid)

    if isinstance(clusterspace, str):
        if clusterspace.lower() == 'pca':
            keys = list(df.keys())
            coord_keys = [key for key in keys if key.startswith('pred_ccoords')]
            N_coords = len(coord_keys)
            coords = []
            for j in range(N_coords):
                coords.append(df[coord_keys[j]])
            coords = np.stack(coords, axis=-1)
            pca = PCA(n_components=3)
            pca_coords = pca.fit_transform(coords)
            pca_x = pca_coords[:, 0]
            pca_y = pca_coords[:, 1]
            pca_z = pca_coords[:, 2]
            df['pca_x'] = pca_x
            df['pca_y'] = pca_y
            df['pca_z'] = pca_z

    print(ids)
    for i in ids:
        if truth:
            df_i = df[df.truthHitAssignementIdx == i]
        else:
            df_i = df[df.pred_sid == i]
        if not clusterspace:
            x = df_i.recHitZ
            y = df_i.recHitY
            z = df_i.recHitX
        elif type(clusterspace) == tuple:
            if not len(clusterspace) == 3:
                print("clusterspace tuple must have length 3")
                exit(1)
            j0, j1, j2 = str(clusterspace[0]), str(clusterspace[1]), str(clusterspace[2])
            x = df_i['pred_ccoords_' + j0]
            y = df_i['pred_ccoords_' + j1]
            z = df_i['pred_ccoords_' + j2]
        elif clusterspace.lower() == 'pca':
            x = df_i['pca_x']
            y = df_i['pca_y']
            z = df_i['pca_z']

        if clusterspace:
            size = np.arctanh(df_i['pred_beta'])
            size *= 5
            size[size < 1.0] = 1.0
        else:
            size = 50 * np.log(df_i.recHitEnergy + 1)
        size[size > 10] = 10
        size[size < 0.1] = 0.1
        color = px.colors.qualitative.Alphabet[i % len(px.colors.qualitative.Alphabet)]
        if i < 0: color = 'black'
        if truth:
            customdata=np.stack((
                df_i['recHitX'], df_i['recHitY'], df_i['recHitZ'],
                df_i['recHitEnergy'], df_i['recHitEta'],
                df_i['truthHitAssignedX'], df_i['truthHitAssignedY'], df_i['truthHitAssignedZ'],
                df_i['truthHitAssignedPIDs'], df_i['t_rec_energy']),
                axis=-1)
        else:
            customdata=np.stack((
                df_i['recHitX'], df_i['recHitY'], df_i['recHitZ'],
                df_i['recHitEnergy'], df_i['recHitEta'],
                df_i['truthHitAssignedX'], df_i['truthHitAssignedY'], df_i['truthHitAssignedZ'],
                df_i['truthHitAssignedPIDs'], df_i['t_rec_energy'],
                df_i['pred_sid'], df_i['pred_energy']),
                axis=-1)
        hovertemplate='<b>RecHit</b>' +\
            '<br>X: %{customdata[0]:.2f}<br>' +\
            'Y: %{customdata[1]:.2f}<br>' +\
            'Z: %{customdata[2]:.2f}<br>' +\
            'Energy: %{customdata[3]:.2f}<br>' +\
            'Eta: %{customdata[4]:.2f}<br>' +\
            'X: %{customdata[5]:.2f}<br>' +\
            'Y: %{customdata[6]:.2f}<br>' +\
            'Z: %{customdata[7]:.2f}<br>' +\
            'PID: %{customdata[8]}<br><br><b>Rec Energy</b><br>%{customdata[9]:.2f}'
        if not truth:
            hovertemplate += '<br><br><b>Prediction</b>' +\
                '<br>sid: %{customdata[10]}<br>' +\
                'energy: %{customdata[11]:.2f}'

        trace_i = go.Scatter3d(
            x = x,
            y = y,
            z = z,
            mode = 'markers',
            marker = dict(
                size = size,
                color = color,
                opacity = 0.8,
                line=dict(width=0),
            ),
            customdata = customdata,
            hovertemplate = hovertemplate,
        )
        fig.add_trace(trace_i)

    return fig


def matched_plot(truth, features, processed_df, showers_df):
    fig = make_figure()
    has_pred = ~np.isnan(showers_df.pred_sid)
    has_truth = ~np.isnan(showers_df.truthHitAssignementIdx)
    matched = np.logical_and(has_pred, has_truth)
    just_pred = np.logical_and(has_pred, ~has_truth)
    just_truth  = np.logical_and(~has_pred, has_truth)
    matched_ids = np.unique(showers_df[matched].pred_sid)
    just_pred_ids = np.unique(showers_df[just_pred].pred_sid)
    just_truth_ids = np.unique(showers_df[just_truth].truthHitAssignementIdx)


    for matched_id in matched_ids:
        color = 'green'
        mask = processed_df.pred_sid == matched_id
        x = features['recHitZ'][mask]
        y = features['recHitY'][mask]
        z = features['recHitX'][mask]
        size = 50 * np.log(features['recHitEnergy'][mask] + 1)

        trace_i = go.Scatter3d(
            x = x,
            y = y,
            z = z,
            mode = 'markers',
            marker = dict(
                size = size,
                color = color,
                opacity = 0.8,
                line=dict(width=0),
            ),
        )
        fig.add_trace(trace_i)

    for just_pred_id in just_pred_ids:
        color = 'red'
        mask = processed_df.pred_sid == just_pred_id
        x = features['recHitZ'][mask]
        y = features['recHitY'][mask]
        z = features['recHitX'][mask]
        size = 50 * np.log(features['recHitEnergy'][mask] + 1)

        trace_i = go.Scatter3d(
            x = x,
            y = y,
            z = z,
            mode = 'markers',
            marker = dict(
                size = size,
                color = color,
                opacity = 0.8,
                line=dict(width=0),
            ),
        )
        fig.add_trace(trace_i)

    for just_truth_id in just_truth_ids:
        color = 'blue'
        mask = truth['truthHitAssignementIdx'] == just_truth_id
        x = features['recHitZ'][mask]
        y = features['recHitY'][mask]
        z = features['recHitX'][mask]
        size = 50 * np.log(features['recHitEnergy'][mask] + 1)

        trace_i = go.Scatter3d(
            x = x,
            y = y,
            z = z,
            mode = 'markers',
            marker = dict(
                size = size,
                color = color,
                opacity = 0.8,
                line=dict(width=0),
            ),
        )

        fig.add_trace(trace_i)

    return fig



def prediction_to_plot(df, id=0):
    return


if __name__ == '__main__':

    INPUTFILE = sys.argv[1]
    OUTPUTDIR = sys.argv[2]

    if HGCALML is None and INPUTFILE.endswith('.djcdc'):
        print("HGCALML not set, cannot work with .djcdc files")
        sys.exit(1)

    if INPUTFILE.endswith('.djcdc'):
        df = djcdc_to_dataframe(INPUTFILE, 10)
    elif INPUTFILE.endswith('.bin.gz'):
        with gzip.open(INPUTFILE, 'rb') as f:
            truth, features = pickle.load(f)
        features = features.drop(columns=['event_id'])
        df = pd.concat([features, truth], axis=1)
    else:
        print("Unknown file type")
        sys.exit(1)

    print(df.head())
    columns = df.columns
    for c in columns: print(c)

    for i in range(10):
        fig = dataframe_to_plot(df, id=i)
        fig.write_html(os.path.join(OUTPUTDIR, f'event_{i}.html'))

