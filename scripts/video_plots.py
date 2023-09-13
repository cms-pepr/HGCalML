import os
import sys
import gzip
import pickle
import numpy as np
import plotly
import plotly.graph_objects as go

SELECTION = [    1,     2,     3,     4,     5,     6,     7,     8,     9,
          10,    11,    12,    13,    14,    15,    16,    17,    18,
          19,    20,    21,    22,    23,    24,    25,    26,    27,
          28,    29,    30,    31,    32,    33,    34,    35,    36,
          37,    38,    39,    40,    41,    42,    43,    44,    45,
          46,    47,    48,    49,    50,    51,    53,    55,    57,
          59,    61,    63,    65,    67,    69,    71,    73,    75,
          77,    79,    81,    83,    85,    87,    89,    91,    93,
          95,    97,    99,   109,   134,   159,   184,   209,   234,
         259,   284,   309,   334,   359,   384,   409,   434,   459,
         484,   509,   534,   559,   584,   619,   719,   819,   919,
        1019,  1119,  1219,  1319,  1419,  1519,  1619,  1719,  1819,
        1919,  2019,  2119,  2219,  2319,  2419,  2519,  2619,  2719,
        2819,  2919,  3019,  3119,  3219,  3319,  3419,  3519,  3619,
        3819,  4019,  4219,  4419,  4619,  4819,  5019,  5219,  5419,
        5619,  5819,  6019,  6219,  6419,  6619,  6819,  7019,  7219,
        7419,  7619,  7819,  8019,  8219,  8419,  8619,  9119,  9619,
       10119, 10619, 11119, 11619, 12119, 12619, 13119, 13619, 14119,
       14619, 15119, 15619, 16119, 16619, 17119, 17619, 18119, 18619,
       19619, 20619, 21619, 22619, 23619, 24619, 25619, 26619, 27619,
       28619, 29619, 30619, 31619, 32619, 33619]

EVENT_ID = 0
BASEPATH = "/mnt/ceph/users/pzehetner/Paper/predictions/0913_video_small/"
IMAGEPATH = "/mnt/home/pzehetner/Connecting-The-Dots/Images"

for i, sel in enumerate(SELECTION):
    if i % 10 == 0:
        print(f"Processing {i}/{len(SELECTION)}")

    path = os.path.join(BASEPATH, f"batch_{sel}/pred_00000.bin.gz"))
    if not os.path.exists(path):
        print(f"File {path} does not exist")
        continue

    phi = 2 * np.pi * i / 100

    with gzip.open(path, "rb") as f:
        data = pickle.load(f)[EVENT_ID]
        beta = data[2]['pred_beta'][:,0]
        sid = data[1]['truthHitAssignementIdx'][:,0]
        x = data[2]['pred_ccoords'][:,0]
        y = data[2]['pred_ccoords'][:,1]
        z = data[2]['pred_ccoords'][:,2]

    fig = go.Figure()


    trace = go.Scatter3d(
        x=beta_cluster[:, 2],
        y=beta_cluster[:, 3],
        z=beta_cluster[:, 4],
        mode='markers',
        marker=dict(
            size=20*beta,
            color=sid,
            colorscale='viridis',
            opacity=0.8,
            line=dict(
                width=0,
            )
        )
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(
            x=2.0 * np.cos(phi),
            y=2.0 * np.sin(phi),
            z=1.0)
    )

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        scene=dict(
            xaxis=dict(range=[-3, 3], autorange=False),
            yaxis=dict(range=[-3, 3], autorange=False),
            zaxis=dict(range=[-3, 3], autorange=False),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        scene_camera=camera
    )


    label = str(sel).zfill(6)
    fig.add_trace(trace)
    fig.update_layout(scene_camera=camera)
    # turn figure to image
    filename = os.path.join(IMAGEPATH, f"event_{EVENT_ID}_batch_{sel}.png")
    fig.write_image(filename)

print("DONE")
