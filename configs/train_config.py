from easydict import EasyDict

cfg = EasyDict()

cfg.device = 'cpu'  # 'cpu'
cfg.workers = 2
cfg.batch_size = 32
cfg.image_size = 64
# cfg.nc = 1
# cfg.nz = 100
# cfg.ngf = 64
# cfg.ndf = 64
cfg.epochs = int(5e9)
cfg.lr = 0.0001
cfg.w_gp = 10

cfg.use_layer_norm = True
cfg.leaky_relu_param = 0.2
cfg.nc = 3
cfg.nz = 100
cfg.ngf = 64
cfg.ndf = 64

cfg.penalty_coef = 10  # lambda
cfg.n_critic = 5
cfg.alpha = 0.0001
cfg.beta1 = 0
cfg.beta2 = 0.9

cfg.log_metrics = False
cfg.experiment_name = 'with_layer_norm'

# cfg.evaluate_on_train_set = True
# cfg.evaluate_before_training = True

cfg.load_saved_model = True
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.epoch_to_load = 6670
cfg.save_model = False
cfg.epochs_saving_freq = 10
