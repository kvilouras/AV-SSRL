#!/bin/bash

if [ $# -eq 5 ]
then
  youtube-dl --ffmpeg-location $4 --cookies $5 --external-downloader ffmpeg --external-downloader-args "-ss $(date -d@$2 -u +%H:%M:%S) -t 00:00:10" -f 'best' --merge-output-format mp4 https://www.youtube.com/watch?v=$1 -o ${3}${1}_${2}'.%(ext)s'
else
  youtube-dl --ffmpeg-location $4 --external-downloader ffmpeg --external-downloader-args "-ss $(date -d@$2 -u +%H:%M:%S) -t 00:00:10" -f 'best' --merge-output-format mp4  https://www.youtube.com/watch?v=$1 -o ${3}${1}_${2}'.%(ext)s'
fi
