import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np


class AdeDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        if for_metrics or opt.phase == "test":
            self.load_size = 512
            opt.load_size = 512
        else:
            self.load_size = 572
            opt.load_size = 572

        opt.crop_size = 512
        self.crop_size = 512

        opt.label_nc = 30
        opt.contain_dontcare_label = True
        opt.semantic_nc = 31 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.phase = opt.phase
        if opt.phase == 'test':
            self.test_all = opt.test_all
        self.dataroot = opt.dataroot
        self.no_flip = opt.no_flip
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        mode = "validation" if (self.phase == "test" and not self.test_all) or self.for_metrics else "training"
        path_img = os.path.join(self.dataroot, "images", mode)
        path_lab = os.path.join(self.dataroot, "annotations", mode)
        img_list = os.listdir(path_img)
        lab_list = os.listdir(path_lab)
        img_list = [filename for filename in img_list if ".png" in filename or ".jpg" in filename]
        lab_list = [filename for filename in lab_list if ".png" in filename or ".jpg" in filename]
        images = sorted(img_list)
        labels = sorted(lab_list)
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0], '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        width, height = image.size
        new_width, new_height = width, height
        if height >= width != self.load_size:
            new_width = self.load_size
            new_height = int((new_width / width) * height)
            image = TR.functional.resize(image, (new_height, new_width), Image.BICUBIC)
            label = TR.functional.resize(label, (new_height, new_width), Image.NEAREST)
        elif width >= height != self.load_size:
            new_height = self.load_size
            new_width = int((new_height / height) * width)
            image = TR.functional.resize(image, (new_height, new_width), Image.BICUBIC)
            label = TR.functional.resize(label, (new_height, new_width), Image.NEAREST)

        if not(self.crop_size == new_width and self.crop_size == new_height):
            # crop
            crop_x = random.randint(0, np.maximum(0, new_width - self.crop_size))
            crop_y = random.randint(0, np.maximum(0, new_height - self.crop_size))
            image = image.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))
            label = label.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))

        # flip
        if not (self.phase == "test" or self.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
