#!/bin/bash
docker run -d -it --name ML4Reco \
-v /home/philipp/Data/Hackathon:/root/Data \
-v "$(pwd)":/root/HGCalML \
-v "/home/philipp/Data/pca-networks/":/root/pca-networks \
-v /home/philipp/Data/Hackathon/ml4reco_sql.config:/root/private/ml4reco_sql.config \
5a0776004b28