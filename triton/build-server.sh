#!/bin/bash

docker build -t local/tritonserver:20.08-tfpepr-py3 -f Dockerfile.oc-server .
