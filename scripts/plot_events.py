"""
Small script to quickly plot some events
"""
import pdb
import os
import sys
import matplotlib.pyplot as plt
from visualize_event import djcdc_to_dataframe, dataframe_to_plot


N_EVENTS = 10

djcdc_path = sys.argv[1]
assert os.path.exists(djcdc_path)
assert djcdc_path.endswith('.djcdc')

event_df = djcdc_to_dataframe(djcdc_path, N_EVENTS)
pdb.set_trace()

for i in range(N_EVENTS):
    fig = dataframe_to_plot(event_df, id=i, truth=True)
    fig.write_html(f'event_{i}.html')
