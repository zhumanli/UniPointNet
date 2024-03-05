# UniPointNet
### A Self-supervised Keypoint Detection Framework for Arbitrary Object Shapes

![Image text](https://github.com/zhumanli/UniPointNet/blob/main/imgs/UniPointNet.png)
We propose UniPointNet which can detect keypoints for arbitrary objects. UniPointNet is designed for object keypoint detection in HOI detection. We employ the self-supervised keypoints learning framework of AutoLink. While AutoLink was proposed to learn keypoints for single object classes, our goal is to detect keypoints across all classes present in the HOI task. To this end, we make two key changes to AutoLink. First, we feed object segmentation masks into the network instead of RGB images. This eliminates the appearance variations across different object classes, simplifying their appearance distribution. As a result, the network can focus on learning object shapes and structures. Second, instead of using an individual edge graph with shared graph weight to align all samples, we opt for a set of edge graphs with different graph weights, aligning samples within their respective clusters. This design accommodates object masks with significant variations, thus allowing the network to detect keypoints across a diverse range of object categories.

# Setup
### Setup environment
```
conda create -n unipointnet python=3.8
conda activate unipointnet
pip install -r requirements.txt
```

### Download dataset
We use object masks from the COCO dataset to train this framework. You can download our processed object masks from [COCO_masks](https://github.com/zhumanli/UniPointNet).

### Download pre-trained models
The pre-trained models can be downloaded from [Google Drive](https://github.com/zhumanli/UniPointNet).

# Testing
To qualitatively test the model, you can run
```
python test.py
```
<img src="https://github.com/zhumanli/UniPointNet/blob/main/imgs/QualitativeResults.png" width="500">

# Training 
To train our model on COCO_masks, run
```
python train.py --n_parts 16 --missing 0.9 --block 16 --thick 2.5e-3 --sklr 512
```

# Acknowledgements
We would like to express our gratitude to the open-source project [AutoLink](https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints) and its contributors since our framework is heavily built on it.
