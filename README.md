# AV-SSRL
MSc Thesis "Audio-Visual Self-Supervised Representation Learning in-the-wild"

## Pre-trained models
We provide checkpoints for models pre-trained on a subset of [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) with 50,000 videos. The former method refers to *Cross-modal Instance Discrimination (xID)*, whereas the latter is based on the recently proposed *VICReg* method.
| Method       | Checkpoint (100 epochs)  |
| ------------- |:-------------:|
| xID      | [download link](https://drive.google.com/uc?export=download&amp;id=1Lc3kK09d4fbhrCKSd1Zx7Mg1BZv-eElw&amp;confirm=t&amp;uuid=dbff4a4f-0615-4358-a767-97e9b9017b57) |
| VICReg      | [download link](https://drive.google.com/uc?export=download&amp;id=1BmZ--xooN0xW3BWVUxS6RUsDTumDzBRa&amp;confirm=t&amp;uuid=e6f1516d-2b96-4b8c-aec6-ba27d630ad53) |


## Self-supervised pre-training
To train a model using *xID* method run the following (assuming that DDP strategy is used):
```python
python3 main-ssl.py configs/VGGSound-N1024.yaml --multiprocessing-distributed
```
For *VICReg* method, run:
```python
python3 main-vicreg.py configs/VGGSound-VICReg.yaml --multiprocessing-distributed
```

To avoid data parallelism, discard `--multiprocessing-distributed` argument and set the `--gpu` argument on either of the aforementioned scripts to a specific id (e.g. 0 for the first GPU device).


## Linear classification
For this experiment, run the following (e.g. for UCF-101 dataset and model pre-trained using *xID* method):
```python
python3 eval-action-recg-linear.py configs/ucf/8at16-linear.yaml configs/VGGSound-N1024.yaml --distributed
```
**Note that this script does not yet support multi-node evaluation.**

Final results on both [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) datasets are shown in the following table:

| Method       | Top-1 Acc. (UCF-101) | Top-5 Acc. (UCF-101) | Top-1 Acc. (HMDB-51) | Top-5 Acc. (HMDB-51) |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| xID | 51.20% | 80.91% | 28.08% | 61.29% |
| VICReg | 39.75% | 71.30% | 21.85% | 52.69% |

## Fine-tuning
For this experiment, run the following (e.g. for HMDB-51 dataset and model pre-trained using *VICReg* method):
```python
python3 eval-action-recg.py configs/hmdb51/8at16-fold1.yaml configs/VGGSound-VICReg.yaml --distributed
```
**Note that this script does not yet support multi-node evaluation.**

Final results on both [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) datasets are shown in the following table:

| Method       | Top-1 Acc. (UCF-101) | Top-5 Acc. (UCF-101) | Top-1 Acc. (HMDB-51) | Top-5 Acc. (HMDB-51) |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| xID | 73.22% | 92.78% | 42.85% | 73.69% |
| VICReg | 59.53% | 85.94% | 34.65% | 68.96% |

## Concept Generalization
In this experiment, we test the generalization performance of self-supervised models on data belonging to unknown classes (i.e. classes not found in the pre-training dataset). To perform the split on the so-called *seen* and *unseen* concepts, please use the `label_similarities.ipynb` notebook. Based on our results, you can find the set of *unseen* concepts for UCF-101 and HMDB-51 respectively in `datasets/rest_classes/` directory.

To perform this experiment, run the following (e.g. for *xID* model and UCF-101 dataset with 20% of training data per class for tuning the linear classifier):
```python
python3 eval-action-recg-linear.py configs/ucf/8at16-linear.yaml configs/VGGSound-N1024.yaml --distributed --few-shot-ratio 0.2 --use-rest-classes
```

Final results are depicted in the following plots:


## References
- [AVID-CMA repository](https://github.com/facebookresearch/AVID-CMA)
- [VICReg repository](https://github.com/facebookresearch/vicreg)
