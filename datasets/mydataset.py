import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms import transforms as transforms
import torchvision
import cv2
from datasets.image_encoding import extract_features

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

centroids = torch.load('datasets/cluster_centroids_128.pt')
random.seed(1)

def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.
    """
    return img.any(axis=-1).sum()

def get_img_label(img_path):

    img_feature = extract_features(img_path)
    dis = np.sqrt(np.sum((centroids - img_feature)**2, axis=1))
    label = np.argmin(dis)

    return label

kernel = np.ones((3, 3), np.uint8)
class mydataset(Dataset):
    def __init__(self, data_dir, flag):
        self.data_info = self.get_img_info(data_dir, flag)  
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = np.asarray(Image.open(path_img))
        ori_img = img
        # area_mask = img.shape[0] * img.shape[1] - count_nonblack_np(img)
        # area_image = 0.001 * img.shape[0] * img.shape[1]
        # area = area_image - area_mask
        # ite = int(area/3) 
        img = cv2.dilate(img, kernel=kernel, iterations=12)
        img = Image.fromarray(img).convert('RGB')    

        if self.transform is not None:
            img = self.transform(img) 
            ori_img = Image.fromarray(ori_img).convert('RGB')   
            ori_img = self.transform(ori_img)

        sample = {'img': img, 'ori_img': ori_img, 'label': label}
        return sample

    def __len__(self):
        return len(self.data_info)
    
    @staticmethod
    def get_img_info(data_dir, flag):
        data_info = list()
        if flag=='train' or flag=='val':
            for root, dirs, _ in os.walk(data_dir):
                for sub_dir in dirs:
                    img_names = os.listdir(os.path.join(root, sub_dir))
                    img_names = list(filter(lambda x: x.endswith(('.png', '.jpg')), img_names))

                    for i in range(len(img_names)):
                        img_name = img_names[i]
                        path_img = os.path.join(root, sub_dir, img_name)
                        label = int(sub_dir)
                        data_info.append((path_img, int(label)))
        else: 
            img_names = os.listdir(data_dir)
            img_names = list(filter(lambda x: x.endswith(('.png', '.jpg')), img_names))

            for i in range(len(img_names)):
                img_name = img_names[i]
                path_img = os.path.join(data_dir, img_name)
                label = get_img_label(path_img)
                data_info.append((path_img, int(label)))
        
        return data_info

