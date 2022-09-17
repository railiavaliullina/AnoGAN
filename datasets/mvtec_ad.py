import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import cv2
from PIL import Image
from enum import Enum, auto
from configs.train_config import cfg as train_cfg

import config


class DatasetType(Enum):
    TRAIN = auto()
    TEST = auto()


class DataClass(Enum):
    Bottle = auto()
    Cable = auto()
    Capsule = auto()
    Carpet = auto()
    Grid = auto()
    Hazelnut = auto()
    Leather = auto()
    Metal_Nut = auto()
    Pill = auto()
    Screw = auto()
    Tile = auto()
    Toothbrush = auto()
    Transistor = auto()
    Wood = auto()
    Zipper = auto()


class MVTEC_AD(Dataset):
    r"""
    https://www.mvtec.com/company/research/datasets/mvtec-ad
    """

    NORMAL_OBJECTS_NAME = 'good'

    def __init__(self, dataset_type, data_type, data_path=config.DATA_PATH):

        self.dataset_type = dataset_type

        data_path = data_path / data_type.name.lower()
        images_path = data_path / dataset_type.name.lower()
        masks_path = (data_path / 'ground_truth')
        self.class_names = [MVTEC_AD.NORMAL_OBJECTS_NAME] + [p.stem for p in images_path.glob('**/')
                                                             if (p.is_dir() and p.stem != MVTEC_AD.NORMAL_OBJECTS_NAME)]

        self.images = []
        self.labels = []
        self.masks = []

        for cls, name in enumerate(self.class_names):
            images_fnames = [p for p in (images_path / name).glob('*.png')]
            for img_fn in images_fnames:
                self.labels.append(cls)
                self.images.append(self.apply_aug(Image.open(str(img_fn))))
                self.masks.append(0)  # cv2.imread(str(masks_path / f'{img_fn.stem}_mask.png'))

    @staticmethod
    def apply_aug(x):
        transform = transforms.Compose([
            transforms.Resize(train_cfg.image_size),
            transforms.ToTensor(),
        ])
        return transform(x)

    def __len__(self):
        return len(self.labels)

    @property
    def num_of_classes(self):
        return len(self.class_names)

    def class2name(self, cls: int):
        return self.class_names[cls]

    def name2class(self, name: str):
        return self.class_names.index(name)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.masks[idx]
