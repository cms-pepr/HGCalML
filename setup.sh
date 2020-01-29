#!/bin/bash

cd $HGCALML

git submodule init
git submodule update

#compile submodule
cd RaggedKnn && \
    git checkout 6214a261129417f41089b9373d1c0c9acc0d2ca5 && \
    wget https://codeload.github.com/NVlabs/cub/zip/1.8.0 -O cub.zip && \
    unzip cub.zip && \
    mv cub-1.8.0 cub &&\
    chmod +x tiny2.sh && \
    ./tiny2.sh
    
#add symlinks
cd $HGCALML/modules
ln -s $HGCALML/RaggedKnn/ragged_knn_kernel.so
ln -s $HGCALML/RaggedKnn/python/rknn_op.py