#!/bin/bash

wget http://storage.googleapis.com/thumos14_files/UCF101_videos.zip
unzip -q UCF101_videos.zip
rm UCF101_videos.zip
mv UCF101/ data/

mkdir UCF101/
mv data UCF101

cd UCF101/
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip -q UCF101TrainTestSplits-RecognitionTask.zip
rm UCF101TrainTestSplits-RecognitionTask.zip