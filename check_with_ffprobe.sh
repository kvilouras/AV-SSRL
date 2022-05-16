#!/bin/bash

dur=$(ffprobe -i $1 -show_streams -select_streams v -loglevel error | grep -w "duration" | cut -d "=" -f2)
thresh=12.0
if [ 1 -eq "$(echo "${dur} > ${thresh}" | bc)" ]
then
  echo 'True'  # must be deleted
else
  echo 'False'
fi
