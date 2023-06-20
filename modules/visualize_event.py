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

# get environment variable 'HGCALML'
try:
    HGCALML = os.environ['HGCALML']
    from DeepJetCore.DataCollection import DataCollection
    from DeepJetCore.dataPipeline import TrainDataGenerator
except KeyError:
    HGCALML = None
    print("HGCALML not set, relying on gzip/pickle")


def dictlist_to_dataframe(dictlist):
    """
    Converta a list of dictionaries to a pandas dataframe
    """
    full_df = pd.DataFrame()
    for i in range(len(dictlist)):
        df = pd.DataFrame()
        for key, value in dictlist[i].items():
            if key in ['row_splits']:
                continue
            if len(value.shape) == 1:
                continue
            if value.shape[1] > 1:
                for j in range(value.shape[1]):
                    df[key + '_' + str(j)] = value[:, j]
            else:
                df[key] = value[:, 0]
        df['event_id'] = i * np.ones_like(df.shape[0])
        full_df = pd.concat([full_df, df])
    return full_df


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
        print(len(data))
        print(td.createFeatureDict(data)['recHitX'].shape)
        truth_list.append(td.createTruthDict(data))
        feature_list.append(td.createFeatureDict(data))

    features = dictlist_to_dataframe(feature_list)
    truth = dictlist_to_dataframe(truth_list)
    features = features.drop(columns=['event_id'])
    output_df = pd.concat([features, truth], axis=1)
    return output_df


def dataframe_to_plot(df, id=0, truth=True):
    df = df[df.event_id == id]
    size = 100 * np.log(df.recHitEnergy + 1)
    # change sizes bigger than 5 to 5
    size[size > 10] = 10
    size[size < 0.1] = 0.1
    print(np.min(size), np.median(size), np.mean(size), np.max(size))

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
    if truth:
        ids = np.unique(df.truthHitAssignementIdx)
    else:
        ids = np.unique(df.pred_sid)
    print(ids)

    for i in ids:
        if truth:
            df_i = df[df.truthHitAssignementIdx == i]
        else:
            df_i = df[df.pred_sid == i]
        x = df_i.recHitZ
        y = df_i.recHitY
        z = df_i.recHitX
        size = 50 * np.log(df_i.recHitEnergy + 1)
        size[size > 10] = 10
        size[size < 0.1] = 0.1
        # get a random discrete color from color palette
        color = np.random.choice(px.colors.qualitative.Alphabet)
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

