#!/bin/bash

mkdir hmdb51/
mkdir hmdb51/videos/
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hmdb51_org.rar hmdb51/videos
cd hmdb51/videos
ls -1 | sed -e 's/\.rar$//' > classNames.txt
mv classNames.txt ../
for filename in ./*.rar; do
    unrar e $filename
    rm $filename
done
cd ../../
rm hmdb51_org.rar

cd hmdb51/
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
unrar x test_train_splits.rar
mv testTrainMulti_7030_splits splits
rm test_train_splits.rar
