import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import io as tvio
from torchvision.transforms import functional as F, InterpolationMode
from PIL import Image
import numpy as np
import os

class SegmentationDataset(Dataset):
    def __init__(self, file_path, image_folder, label_folder, superpixel_folder, size=(128,128), num_classes=20, crop=False):
        self.file_path = file_path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.superpixel_folder = superpixel_folder

        self.size = size
        self.num_classes = num_classes

        self.crop = crop
        self.random_crop = transforms.RandomCrop(self.size)

        self.large_sample = transforms.Resize((self.size[0]*2,self.size[1]*2), interpolation=InterpolationMode.NEAREST)

        self.image_transform = transforms.Compose([
            transforms.Resize(self.size, antialias=True),
        ])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.size, antialias=False, interpolation=InterpolationMode.NEAREST)
        ])

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(script_dir, self.file_path)
        self.image_folder = os.path.join(script_dir, self.image_folder)
        self.label_folder = os.path.join(script_dir, self.label_folder)
        self.superpixel_folder = os.path.join(script_dir, self.superpixel_folder)

        # Open the file and read its contents
        with open(self.file_path, 'r') as file:
            # Read all lines in the file
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_name = self.lines[idx].strip() + '.jpg'  # Assuming images have .jpg extension; adjust if needed
        image_path = os.path.join(self.image_folder, image_name)
        mask_name = self.lines[idx].strip() + '.png'
        mask_path = os.path.join(self.label_folder, mask_name)
        # superpixel_name = self.lines[idx].strip() + '.png'
        # superpixel_path = os.path.join(self.superpixel_folder, mask_name)

        # Check if the image file exists before attempting to open it
        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = tvio.read_image(image_path).float()
            gt = tvio.read_image(mask_path)
            if self.crop:
                image = self.large_sample(image)
                gt = self.large_sample(gt)
                total = torch.cat((image, gt),dim = 0)
                total = self.random_crop(total)
                image = total[0:3,:,:]
                gt = total[3,:,:]
            image = self.image_transform(image)
            x = self.normalize(image)
            gt = self.mask_transform(gt)
            gt = gt.squeeze()
            gt[(gt < 255) & (gt > self.num_classes - 1)] = 0
        else:
            print(f"Warning: Image file not found - {image_path}")
            # Return a placeholder if the file is not found
            image = torch.zeros((3,)+self.size, dtype=torch.float32)

            print(f"Warning: Image file not found - {mask_path}")
            # Return a placeholder if the file is not found
            gt = torch.zeros(self.size, dtype=torch.long)

        return {"image": image.float(), "mask": gt, 'x': image.float()}


class AttackDataset(Dataset):
    def __init__(self, file_path, image_folder, label_folder, ispng=False):
        self.file_path = file_path
        self.image_folder = image_folder
        self.label_folder = label_folder

        self.num_classes = 21
        
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.totensor = transforms.ToTensor()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(script_dir, self.file_path)
        self.image_folder = os.path.join(script_dir, self.image_folder)
        self.label_folder = os.path.join(script_dir, self.label_folder)

        self.pngjpg = '.png' if ispng else '.jpg'
        # Open the file and read its contents
        with open(self.file_path, 'r') as file:
            # Read all lines in the file
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_name = self.lines[idx].strip() + self.pngjpg  # Assuming images have .jpg extension; adjust if needed
        image_path = os.path.join(self.image_folder, image_name)
        mask_name = self.lines[idx].strip() + '.png'
        mask_path = os.path.join(self.label_folder, mask_name)
        # superpixel_name = self.lines[idx].strip() + '.png'
        # superpixel_path = os.path.join(self.superpixel_folder, mask_name)

        # Check if the image file exists before attempting to open it
        if os.path.exists(image_path) and os.path.exists(mask_path):
            input_image = Image.open(image_path)
            unnormal = self.totensor(input_image)
            gt = self.totensor(Image.open(mask_path))

            input_image = input_image.convert("RGB")
            input_tensor = self.preprocess(input_image)
            
        else:
            raise NotImplementedError

        return {"image": input_tensor.float(), "gt": gt, "unnormal": unnormal}