#!/usr/bin/env python3

import numpy as np
import pickle
import sys
from ragged_plotting_tools import make_plots_from_object_condensation_clustering_analysis





with open(sys.argv[1], 'rb') as f:
    dataset_analysis_dict = pickle.load(f)

make_plots_from_object_condensation_clustering_analysis(sys.argv[2], dataset_analysis_dict)

