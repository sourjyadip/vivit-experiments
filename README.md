# vivit-experiments
This repository includes code to run Frame Average Pooling method on the ViViT model.

## Instructions

### Download the Kinetics-400 dataset
The dataset can be downloaded from https://github.com/cvdfoundation/kinetics-dataset.

### Set up the environment
Create a conda python environment (python=3.10) and install the dependencies.
```
pip install -r requirements.txt
```
### Change the file paths in the src/train.py file.

### Run the training code, which also gives results on the test set after every epoch.
```
python src/train.py
```
## System Requirements
- NVIDIA A40 GPU (48GB) 
- 100 GB free hard disk space
