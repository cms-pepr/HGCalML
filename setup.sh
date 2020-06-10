#!/bin/bash

cd $HGCALML


rm -f $HGCALML/modules/ragged_knn_kernel.so
rm -rf  $HGCALML/RaggedKnn/cub-1.8.0 $HGCALML/RaggedKnn/cub
#compile submodule
cd RaggedKnn && \
    wget https://codeload.github.com/NVlabs/cub/zip/1.8.0 -O cub.zip && \
    unzip cub.zip && \
    mv cub-1.8.0 cub &&\
    chmod +x tiny2.sh && \
    ./tiny2.sh
    
#add symlinks
cd $HGCALML/modules
ln -s $HGCALML/RaggedKnn/ragged_knn_kernel.so
ln -s $HGCALML/RaggedKnn/python/rknn_op.py

cd compiled
make -j
cd $HGCALML
