import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np

cm = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]] + [[0,0,0],]*234 + [[255,255,255],]

VOC_COLORMAP = ListedColormap(cm)

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv_monitor']

def IoU(pred, target, num_classes):
    iou = {}
    for i in range(1, num_classes):
        inter = ((pred==i) & (target==i)).float().sum()
        uni = ((pred==i) | (target==i)).float().sum()
        if uni != 0:
            iou[VOC_CLASSES[i]] = (inter / uni).cpu().detach().numpy()
    return iou

def plot_segmentation(mask, path):
    plt.clf()
    plt.axis('off')
    plt.imshow(mask, cmap=VOC_COLORMAP, vmin=0, vmax=255)
    plt.savefig(path)

def plot_superpixel(mask, path):
    mask = mask.squeeze()
    plt.clf()
    plt.axis('off')
    plt.imshow(mask)
    plt.savefig(path)

def get_boundaries(labels, threshold=0):
    ht,wd = labels.shape
    boundary = np.zeros_like(labels)
    for y in range(1,ht-1):
        for x in range(1,wd-1):
            if np.abs(labels[y,x-1]-labels[y,x+1]) > 1 or np.abs(labels[y-1,x]-labels[y+1,x]) > threshold:
                boundary[y,x] = 255
    return boundary

def plot_png(img, path):
    Image.fromarray(img.astype(np.uint8)).save(path)

def plot_boundary(mask, path, threshold=0):
    Image.fromarray(get_boundaries(mask, threshold).astype(np.uint8)).save(path)