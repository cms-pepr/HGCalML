#!/bin/bash

cd $HGCALML/modules
cd compiled
make -j
cd $HGCALML
git submodule update --init --recursive
