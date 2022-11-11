from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from glob import glob
import random


class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        print(self._image_dir)
        imagelist = glob(self._image_dir + "/*.png")
        for image_path in imagelist:
            gt_path = image_path.replace('image', 'mask')
            self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

        self.transform = transform
        # self._read_img_into_memory()
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]
        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'

class ReFundusSegmentation(Dataset):

    def __init__(self,
                 base_dir,
                 re_base_dir,
                 dataset='fundus',
                 #split='train',
                 redataset='wae_img/Domain3',
                 testid=None,
                 transform=None
                 ):
        self._base_dir = base_dir
        self._re_base_dir = re_base_dir
        self.image_list = []
        #self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []

        self._image_dir = os.path.join(self._base_dir, dataset, 'train/ROIs', 'image')
        self._re_image_dir = os.path.join(self._re_base_dir, redataset)
        print(self._image_dir)
        print(self._re_image_dir)
        imagelist_1 = glob(self._image_dir + "/*.png")
        imagelist_2 = glob(self._re_image_dir + "/*.png")
        for image_path in imagelist_1:
            gt_path = image_path.replace('image', 'mask')
            self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
        for image_path_2 in imagelist_2:
            gt_path_2 = image_path_2.replace(self._re_image_dir, self._image_dir)
            gt_path_2 = gt_path_2.replace('image', 'mask')
            gt_path_2 = gt_path_2.replace('re_', '')
            self.image_list.append({'image': image_path_2, 'label': gt_path_2, 'id': testid})

        self.transform = transform
        # self._read_img_into_memory()
        # Display stats
        print('Number of images in {}: {:d}'.format('new data', len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _target = Image.open(self.image_list[index]['label'])
        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]
        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'


