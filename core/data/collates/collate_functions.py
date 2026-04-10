# -*- coding: utf-8 -*-
import itertools
from collections import Iterable
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import ImageEnhance, ImageFilter
import torch


# Helpers below are used by the official LDP-Net training collate.
identity = lambda x: x

transformtypedict={
    "Brightness": ImageEnhance.Brightness,
    "Contrast": ImageEnhance.Contrast,
    "Sharpness": ImageEnhance.Sharpness,
    "Color": ImageEnhance.Color,
}

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms =[(transformtypedict[k],transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        for i, (transformer,alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i]*2.0-1.0)+1
            out = transformer(out).enhance(r).convert('RGB')
        return out

class PILRandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))
        )


def get_color_distortion(s=0.5):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([rnd_color_jitter, rnd_gray])


class GeneralCollateFunction(object):
    """A Generic `Collate_fn`.

    For finetuning-train.
    """

    def __init__(self, trfms, times):
        """Initialize a `GeneralCollateFunction`.

        Args:
            trfms (list): A list of torchvision transforms.
            times (int): Specify the augment times. (0 or 1 for not to augment)
        """
        super(GeneralCollateFunction, self).__init__()
        self.trfms = trfms
        self.times = times

    def method(self, batch):
        """Apply transforms and augmentations on a batch.

        The images and targets in a batch are augmented by the number of `self.times` and the targets are augmented
        to match the shape of images.

        Args:
            batch (list of tuple): A batch returned by dataset.

        Returns:
            tuple: A tuple of (images, targets), here len(images)=len(targets).
        """
        try:
            images, targets = zip(*batch)

            images = list(
                itertools.chain.from_iterable(
                    [[image] * self.times for image in images]
                )
            )
            images = [self.trfms(image).unsqueeze(0) for image in images]

            targets = list(
                itertools.chain.from_iterable(
                    [[target] * self.times for target in targets]
                )
            )
            targets = [torch.tensor([target]) for target in targets]

            assert len(images) == len(
                targets
            ), "Inconsistent number of images and labels!"

            images = torch.cat(images)

            targets = torch.tensor(targets, dtype=torch.int64)

            return images, targets
        except TypeError:
            raise TypeError(
                "Error, probably because the transforms are passed to the dataset, the transforms should be "
                "passed to the collate_fn"
            )

    def __call__(self, batch):
        return self.method(batch)


class FewShotAugCollateFunction(object):
    """`Collate_fn` for few-shot dataloader.

    For finetuning-val, finetuning-test and meta/metric-train/val/test.
    """

    def __init__(self, trfms, times, times_q, way_num, shot_num, query_num):
        """Initialize a `FewShotAugCollateFunction`.


        Args:
            trfms (list or tuple of list): A torchvision transfrom list of a tuple of 2 torchvision transform list.
            if  `list`, both support and query images will be applied the same transforms, otherwise the 1st one will
            apply to support images and the 2nd one will apply to query images.
            times (int): Augment times of support iamges
            times_q (int ): Augment times of query images
            way_num (int): Few-shot way setting
            shot_num (int): Few-shot shot setting
            query_num (int): Few-shot query setting
        """
        super(FewShotAugCollateFunction, self).__init__()
        try:
            self.trfms_support, self.trfms_query = trfms
        except Exception:
            self.trfms_support = self.trfms_query = trfms
        # self.trfms = trfms
        # Allow different trfms: when single T, apply to S and Q equally;
        # When trfms=(T,T), apply to S and Q separately;
        self.times = 1 if times == 0 else times
        self.times_q = 1 if times_q == 0 else times_q
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.shot_aug = self.shot_num * self.times
        self.query_aug = self.query_num * self.times_q

    def method(self, batch):
        """Apply transforms and augmentations on a **few-shot** batch.

        The samples of query and support are augmented separately.
        For example: if aug_times=5, then 01234 -> 0000011111222223333344444.

        Args:
            batch (list of tuple): A batch returned by a few-shot dataset.

        Returns:
            tuple: a tuple of (images, gt_labels).
        """
        try:
            images, labels = zip(
                *batch
            )  # images = [img_label_tuple[0] for img_label_tuple in batch]  # 111111222222 (5s1q for example)
            images_split_by_label = [
                images[index : index + self.shot_num + self.query_num]
                for index in range(0, len(images), self.shot_num + self.query_num)
            ]  # 111111; 222222 ;
            images_split_by_label_type = [
                [spt_qry[: self.shot_num], spt_qry[self.shot_num :]]
                for spt_qry in images_split_by_label
            ]  # 11111,1;22222,2;  == [shot, query]

            # aug support # fixme: should have a elegant method # 1111111111,1;2222222222,2 # (aug_time = 2 for example)
            for cls in images_split_by_label_type:
                cls[0] = cls[0] * self.times  # aug support
                cls[1] = cls[1] * self.times_q  # aug query

            # flatten and apply trfms
            flat = (
                lambda t: [x for sub in t for x in flat(sub)]
                if isinstance(t, Iterable)
                else [t]
            )
            images = flat(images_split_by_label_type)  # 1111111111122222222222
            # images = [self.trfms(image) for image in images]  # list of tensors([c, h, w])
            images = [
                self.trfms_support(image)
                if index % (self.shot_aug + self.query_aug) < self.shot_aug
                else self.trfms_query(image)
                for index, image in enumerate(images)
            ]  # list of tensors([c, h, w])
            images = torch.stack(images)  # [b', c, h, w] <- b' = b after aug

            # labels
            # global_labels = torch.tensor(labels,dtype=torch.int64)
            # global_labels = torch.tensor(labels,dtype=torch.int64).reshape(self.episode_size,self.way_num,
            # self.shot_num*self.times+self.query_num)
            global_labels = torch.tensor(labels, dtype=torch.int64).reshape(
                -1, self.way_num, self.shot_num + self.query_num
            )
            global_labels = (
                global_labels[..., 0]
                .unsqueeze(-1)
                .repeat(
                    1,
                    1,
                    self.shot_num * self.times + self.query_num * self.times_q,
                )
            )

            return images, global_labels
            # images.shape = [e*w*(q+s) x c x h x w],  global_labels.shape = [e x w x (q+s)]
        except TypeError:
            raise TypeError(
                "Error, probably because the transforms are passed to the dataset, the transforms should be "
                "passed to the collate_fn"
            )

    def __call__(self, batch):
        return self.method(batch)

class LDPFewShotCollateFunction(object):
    """
    LDP-Net official-style episodic training collate.

    Output:
    - batch_views: dict
        - "local_views": list of 6 tensors, each [E, W, S+Q, C, 96, 96]
        - "raw_images": tensor [E, W, S+Q, C, 224, 224]
    - global_labels: tensor [E, W, S+Q]
    """
    def __init__(self, way_num, shot_num, query_num):
        super(LDPFewShotCollateFunction, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        color_transform = [get_color_distortion(),PILRandomGaussianBlur()]

        self.local_transforms = [
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(96,scale=(0.05,0.14)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Compose(color_transform),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
            for _ in range(6)
        ]

        self.raw_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                ImageJitter(dict(Brightness=0.4, Contrast=0.4,Color=0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def method(self, batch):
        images, labels = zip(*batch)

        episode_size = len(images) // (self.way_num * (self.shot_num + self.query_num))

        images = list(images)
        labels = torch.tensor(labels, dtype=torch.int64).reshape(
            episode_size, self.way_num, self.shot_num + self.query_num
        )

        local_views = []
        for trans in self.local_transforms:
            view = [trans(img) for img in images]
            view =torch.stack(view).view(
                episode_size, self.way_num, self.shot_num + self.query_num,3,96,96
            )
            local_views.append(view)

        raw_images = [self.raw_transforms(img) for img in images]
        raw_images = torch.stack(raw_images).view(
            episode_size, self.way_num, self.shot_num + self.query_num,3,224,224
        )

        batch_views = {
            "local_views": local_views,
            "raw_images": raw_images,
        }

        return batch_views,labels

    def __call__(self, batch):
        return self.method(batch)