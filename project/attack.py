import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from data.dataloader import SegmentationDataset, AttackDataset
from torch.utils.data import DataLoader, Subset

import os
import numpy as np
from matplotlib.colors import ListedColormap
from argparse import ArgumentParser

import wandb

def arg_parser():
  parser = ArgumentParser(description='Segmentation Attack Project')
  parser.add_argument('--datetime', type=str, default='20230000')
  parser.add_argument('--runname', type=str, default='JPEGImages')

  args = parser.parse_args()
  return args

def accuracy(pred, target):
    correct = (torch.abs(pred-target) < 1).long().sum().float()
    total = (target < 21).long().sum().float()
    return (correct / total).cpu().detach().numpy()

def IoU(pred, target, num_classes):
    _mIoU = []

    a = ((pred==0) & (target==0)).long().sum().float()
    b = ((pred==0) | (target==0)).long().sum().float()
    iou = (a / (b + 1e-7)).cpu().detach().numpy()
    if iou > 0:
        _mIoU.append(iou)

    pred[target > num_classes - 1] = 0
    target[target > num_classes - 1] = 0
    for i in range(1, num_classes):
        a = ((pred==i) & (target==i)).long().sum().float()
        b = ((pred==i) | (target==i)).long().sum().float()
        iou = (a / (b + 1e-7)).cpu().detach().numpy()
        if iou > 0:
            _mIoU.append(iou)
    # print(_mIoU)
    return np.average(_mIoU)

def main(args):
    runname = args.runname
    wandb.init(project="semantic_segmentation_attack_project", name=runname)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.eval()

    # Download an example image from the pytorch website

    val_path = 'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

    image_folder = f'VOCdevkit/VOC2012/{runname}'
    ispng = True
    if image_folder == 'VOCdevkit/VOC2012/JPEGImages':
        ispng = False
    label_folder = 'VOCdevkit/VOC2012/SegmentationClass'

    image_size = (256, 256)
    num_classes = 21

    # Create an instance of the CustomDataset
    # train_dataset = AttackDataset(file_path=train_path, image_folder=image_folder, label_folder=label_folder)
    val_dataset = AttackDataset(file_path=val_path, image_folder=image_folder, label_folder=label_folder, ispng=ispng)
    val_limit = torch.arange(40)
    val_dataset = Subset(val_dataset, val_limit)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    # sample execution (requires torchvision)
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    model.to(device)
    
    acc = []
    mIoU = []

    storage_dir = f'storage/attack/{args.datetime}'
    os.makedirs(storage_dir, exist_ok=True)
    colors = ListedColormap(np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]) / 255.0)
    log_dict = {'input':[], 'pred': []}
    for batch_idx, input_batch in enumerate(val_loader):
        print(f'BATCH NUM: {batch_idx}')
        if torch.cuda.is_available():
            input_image = input_batch['image'].to(device)

        # move the input and model to GPU for speed if available

        with torch.no_grad():
            output = model(input_image)['out'][0]
        output_predictions = output.argmax(0)
        gt = input_batch['gt'][0,0].to(device)
        
        batch_acc = accuracy(output_predictions, gt)
        batch_mIoU = IoU(output_predictions, gt, num_classes)
        print(f'BATCH ACC: {batch_acc}')
        print(f'BATCH mIoU: {batch_mIoU}')
        acc.append(batch_acc)
        mIoU.append(batch_mIoU)

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((input_image[0][0].shape[1],input_image[0][0].shape[0]))
        x = (input_batch['unnormal'][0].permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)

        Image.fromarray(x).save(f'{storage_dir}/image{batch_idx}.png')
        plt.imsave(f'{storage_dir}/prediction{batch_idx}.png', r, cmap=colors, vmin=0, vmax=20)

        log_dict['input'].append(wandb.Image(f'{storage_dir}/image{batch_idx}.png'))
        log_dict['pred'].append(wandb.Image(f'{storage_dir}/prediction{batch_idx}.png'))

    print('done')
    log_dict['accuracy'] = np.average(acc)
    log_dict['mIoU'] = np.average(mIoU)
    wandb.log(log_dict)

if __name__ == '__main__':
    args = arg_parser()
    main(args)
