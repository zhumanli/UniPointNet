# UniPointNet
## A Self-supervised Keypoint Detection Framework for Arbitrary Object Shapes

![Image text](https://github.com/zhumanli/UniPointNet/blob/main/imgs/UniPointNet.png)

# Setup
## Setup environment
```
conda create -n unipointnet python=3.8
conda activate unipointnet
pip install -r requirements.txt
```

## Download dataset
We use object masks from the COCO dataset to train this framework. You can download our processed object masks from [COCO_masks](https://github.com/zhumanli/UniPointNet).

## Download pre-trained models
The pre-trained models can be downloaded from [Google Drive](https://github.com/zhumanli/UniPointNet).

# Testing
To qualitatively test the model, you can run
'''
python test.py
'''

# Training 
To train our model on COCO_masks, run
'''
python train.py --n_parts 16 --missing 0.9 --block 16 --thick 2.5e-3 --sklr 512
'''

# Acknowledgements
We would like to express our gratitude to the open-source project [AutoLink](https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints) and its contributors since our framework is heavily built on it.
