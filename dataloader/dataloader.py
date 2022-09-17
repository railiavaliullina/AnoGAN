import torch
from configs.train_config import cfg as train_cfg
from datasets.mvtec_ad import MVTEC_AD, DatasetType, DataClass


def get_dataloaders():
    """
    Initializes train, test datasets and gets their dataloaders.
    :return: train and test dataloaders
    """
    train_dataset = MVTEC_AD(DatasetType.TRAIN, DataClass.Capsule)
    test_dataset = MVTEC_AD(DatasetType.TEST, DataClass.Capsule)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size, drop_last=True, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=train_cfg.batch_size)
    return train_dl, test_dl
