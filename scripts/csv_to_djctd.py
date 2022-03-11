from datastructures.TrainData_TrackML import TrainData_TrackML
import os
import sys

print("All imports done...")

foldernames = [
    '/eos/project-d/dshep/TrackML/extracted/train_1',
    '/eos/project-d/dshep/TrackML/extracted/train_2',
    '/eos/project-d/dshep/TrackML/extracted/train_3',
    '/eos/project-d/dshep/TrackML/extracted/train_4',
    '/eos/project-d/dshep/TrackML/extracted/train_5',
]
outputfoldernames = [
    '/eos/project-d/dshep/TrackML/extracted/train_1/converted3',
    '/eos/project-d/dshep/TrackML/extracted/train_2/converted3',
    '/eos/project-d/dshep/TrackML/extracted/train_3/converted3',
    '/eos/project-d/dshep/TrackML/extracted/train_4/converted3',
    '/eos/project-d/dshep/TrackML/extracted/train_5/converted3',
]

# foldernames = ['/mnt/home/sqasim/ceph/Datasets/TrackML/train_100_events']
# outputfoldernames = ['/mnt/home/sqasim/ceph/Datasets/TrackML/train_100_events/converted']







for foldername, outputfoldername in zip(foldernames, outputfoldernames):

    # foldername = '/eos/project-d/dshep/TrackML/extracted/train_100_events'
    # outputfoldername = '/eos/project-d/dshep/TrackML/extracted/train_100_events/converted2'
    # x =                '/mnt/home/sqasim/ceph/Datasets/toy_calo_nov_17_20/conversion/djc4'


    # t = int(sys.argv[1])

    # with open('files_x.txt') as f:
    #     content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = os.listdir(foldername)
    content = [x.strip() for x in content]

    for i, _filename in enumerate(content):
        # print("Working on ", _filename)
        if not _filename.endswith('-truth.csv'):
            continue
        filename_truth = os.path.join(foldername, _filename)
        filename_hits = os.path.join(foldername, _filename[:-9]+'hits.csv')
        filename_cells = os.path.join(foldername, _filename[:-9]+'cells.csv')
        filename_particles = os.path.join(foldername, _filename[:-9]+'particles.csv')


        outfilename = os.path.join(outputfoldername, _filename[:-9]+'td.djctd')
        TrainData_TrackML()
        td = TrainData_TrackML()
        print("Converting ", filename_truth)
        td.createFromCsvsIntoStandard(filename_truth,filename_hits,filename_cells,filename_particles, outfilename)
        print("Done", i, len(content))

        # print(filename_hits, os.path.exists(filename_hits))
        # print(filename_truth, os.path.exists(filename_truth))
        # print(filename_cells, os.path.exists(filename_cells))
        # print(filename_particles, os.path.exists(filename_particles))


        # outfilename = os.path.join(outputfoldername, os.path.splitext(_filename)[0]+'.djctd')
        # td = TrainData_TrackML()
        # td.combineToySet(filename, outfilename)
        # print("Done")