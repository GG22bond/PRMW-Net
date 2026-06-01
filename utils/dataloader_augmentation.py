import cv2
import os
import numpy as np
import random
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations.pytorch as A

# color
def switch_color(img1, img2):
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2LAB)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2LAB)

    mean1, std1 = cv2.meanStdDev(img1)
    mean2, std2 = cv2.meanStdDev(img2)

    img1 = img1.astype(np.float32)
    img1 = (img1 - mean1[: ,0] ) /(std1[: ,0] + 1e-8) * std2[: ,0] + mean2[: ,0]
    img1 = np.clip(img1, 0 , 255).astype(np.uint8)
    img1 = cv2.cvtColor(img1, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img1.astype(np.uint8))

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations, switch_ratio = 0.0):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.switch_ratio = switch_ratio
        print("switch_ratio:", self.switch_ratio)

        if self.augmentations:
            print('Using RandomRotation, RandomFlip')

            self.img_transform = transforms.Compose([
                transforms.RandomRotation(15, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(15, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()])
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):

        image = self.rgb_loader(self.images[index])
        if random.random() < self.switch_ratio:
            image2 = self.rgb_loader(random.choice(self.images))
            image = switch_color(image, image2)
        gt = self.binary_loader(self.gts[index])

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, switch_ratio = 0, num_workers=4, pin_memory=True, augmentation=False, sampler = None):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentations=augmentation, switch_ratio=switch_ratio)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  sampler=sampler,
                                  shuffle=shuffle)
    return data_loader

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
