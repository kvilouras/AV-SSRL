#!/bin/bash

if [[ $(ffprobe -i $1 -show_streams -select_streams a -loglevel error) ]]; then
    echo "False"  # audio stream exists
else
    echo "True"  # audio stream does not exist!
fi
