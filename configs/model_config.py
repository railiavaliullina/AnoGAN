from easydict import EasyDict
from configs.train_config import cfg as train_cfg

cfg = EasyDict()

cfg.in_features_size = 48758  # 50000
cfg.inner_layers_num = 1
cfg.inner_features_size = 128  # [1024, 256]
cfg.out_features_size = 4

cfg.use_layer_norm = True
cfg.leaky_relu_param = 0.2
cfg.nc = 3
cfg.nz = 100
cfg.ngf = 64
cfg.ndf = 64
cfg.device = train_cfg.device
