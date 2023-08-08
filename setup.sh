#!/bin/bash

cd $HGCALML/modules
cd compiled
make -j4
cd $HGCALML
git submodule update --init --recursive
