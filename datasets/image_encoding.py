import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

resnet = models.resnet50(pretrained=True)

num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 128)

# Define the image transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define a function to load the images and extract features
def extract_features(img_path, resnet=resnet, transform=transform):
    
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    feature = resnet(img)
    feature = feature.detach().numpy()
    return feature

