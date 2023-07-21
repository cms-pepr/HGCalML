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

# get environment variable 'HGCALML'
try:
    HGCALML = os.environ['HGCALML']
    from DeepJetCore.DataCollection import DataCollection
    from DeepJetCore.dataPipeline import TrainDataGenerator
except KeyError:
    HGCALML = None
    print("HGCALML not set, relying on gzip/pickle")
    sys.exit(1)


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
    for c in columns:
        print(c)

    with gzip.open(os.path.join(OUTPUTDIR, 'features.bin.gz'), 'wb') as f:
        pickle.dump(df, f)




