from torch import nn
import torch
import os
from torchvision import utils as vutils
import time
import numpy as np

from dataloader.dataloader import get_dataloaders
from models.GAN import get_model
from utils.logging import Logger
from utils import metrics


class Trainer(object):
    def __init__(self, cfg, model_cfg):
        """
        Class for initializing and performing training procedure.
        :param cfg: train config
        """
        self.cfg = cfg
        self.dl_train, self.dl_test = get_dataloaders()
        self.netG, self.netD = get_model(cfg)
        self.criterion = self.get_criterion()
        self.optimizerD, self.optimizerG = self.get_optimizer()
        self.logger = Logger(self.cfg)
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def get_criterion():
        """
        Gets criterion.
        :return: criterion
        """
        criterion = nn.BCELoss()
        return criterion

    def get_optimizer(self):
        """
        Gets optimizer.
        :return: optimizer
        """
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
        return optimizerD, optimizerG

    def restore_model(self, model, optimizer, net='generator'):
        """
        Restores saved model.
        """
        if self.cfg.load_saved_model:
            print(f'Trying to load checkpoint for net {net} from epoch {self.cfg.epoch_to_load}...')
            try:
                checkpoint = torch.load(self.cfg.checkpoints_dir + f'/{net}_checkpoint_{self.cfg.epoch_to_load}.pth')
                load_state_dict = checkpoint['model']
                model.load_state_dict(load_state_dict)
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step'] + 1
                optimizer.load_state_dict(checkpoint['opt'])
                print(f'Loaded checkpoint for net {net} from epoch {self.cfg.epoch_to_load}.')
            except FileNotFoundError:
                print('Checkpoint not found')
        return model, optimizer

    def save_model(self, model=None, optimizer=None, net='generator'):
        """
        Saves model.
        """
        if self.cfg.save_model and self.epoch % self.cfg.epochs_saving_freq == 0:
            print('Saving current model...')
            state = {
                'model': model.state_dict() if model is not None else None,
                'epoch': self.epoch,
                'global_step': self.global_step,
                'opt': optimizer.state_dict() if optimizer is not None else None
            }
            if not os.path.exists(self.cfg.checkpoints_dir):
                os.makedirs(self.cfg.checkpoints_dir)

            path_to_save = os.path.join(self.cfg.checkpoints_dir, f'{net}_checkpoint_{self.epoch}.pth')
            torch.save(state, path_to_save)
            print(f'Saved {net} model to {path_to_save}.')

    def calculate_anomaly_score(self, x, noise, lambda_=0.1):
        _, x_feature = self.netD(x)
        _, noise_feature = self.netD(noise)
        loss_r = torch.sum(torch.abs(x - noise))
        loss_d = torch.sum(torch.abs(x_feature - noise_feature))
        loss = (1 - lambda_) * loss_r + lambda_ * loss_d
        return loss

    def train(self, train_gan=True, train_z=True):
        """
        Runs training procedure.
        """
        self.start_epoch, self.epoch, self.global_step = 0, -1, 0

        # restore model if necessary
        self.netD, self.optimizerD = self.restore_model(self.netD, self.optimizerD, net='discriminator')
        self.netG, self.optimizerG = self.restore_model(self.netG, self.optimizerG, net='generator')

        if train_gan:
            total_training_start_time = time.time()
            fixed_noise = torch.randn(self.cfg.batch_size, self.cfg.nz, 1, 1, device=self.cfg.device)
            real_label = torch.full((self.cfg.batch_size,), 1., dtype=torch.float, device=self.cfg.device)
            fake_label = torch.full((self.cfg.batch_size,), 0., dtype=torch.float, device=self.cfg.device)

            G_losses, D_losses, D_accs, D_accs_real, D_accs_fake = [], [], [], [], []
            print(f'Starting training...')
            iter_num = len(self.dl_train)
            for epoch in range(self.start_epoch, self.cfg.epochs):
                epoch_start_time = time.time()
                self.epoch = epoch
                print(f'Epoch: {self.epoch}/{self.cfg.epochs}')

                for i, data in enumerate(self.dl_train):
                    self.netD.zero_grad()
                    real_cpu = data[0].to(self.cfg.device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size,), 1, dtype=torch.float, device=self.cfg.device)
                    output, _ = self.netD(real_cpu)
                    output = output.view(-1)
                    real_acc = torch.sum(torch.round(output) == real_label) / self.cfg.batch_size
                    errD_real = self.criterion(output, label)
                    errD_real.backward()
                    D_x = output.mean().item()
                    noise = torch.randn(b_size, self.cfg.nz, 1, 1, device=self.cfg.device)
                    fake = self.netG(noise)
                    label.fill_(0)
                    output, _ = self.netD(fake.detach())
                    output = output.view(-1)
                    fake_acc = torch.sum(torch.round(output) == fake_label) / self.cfg.batch_size

                    acc = ((fake_acc + real_acc) / 2).item()
                    D_accs.append(acc)
                    D_accs_fake.append(fake_acc.item())
                    D_accs_real.append(real_acc.item())
                    errD_fake = self.criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    loss_d = errD_real + errD_fake
                    self.optimizerD.step()
                    D_losses.append(loss_d.item())

                    self.netG.zero_grad()
                    label.fill_(1)
                    output, _ = self.netD(fake)
                    output = output.view(-1)
                    loss_g = self.criterion(output, label)
                    loss_g.backward()
                    G_losses.append(loss_g.item())
                    D_G_z2 = output.mean().item()
                    self.optimizerG.step()

                    if i % 1 == 0:
                        mean_acc_fake = np.mean(D_accs_fake[-10:]) if len(D_accs_fake) > 10 else np.mean(D_accs_fake)
                        mean_acc_real = np.mean(D_accs_real[-10:]) if len(D_accs_real) > 10 else np.mean(D_accs_real)
                        mean_loss_d = np.mean(D_losses[-10:]) if len(D_losses) > 10 else np.mean(D_losses)
                        mean_loss_g = np.mean(G_losses[-10:]) if len(G_losses) > 10 else np.mean(G_losses)

                        print(f'Epoch: {epoch}/{self.cfg.epochs}, iter:{i}/{iter_num}, step: {self.global_step}, '
                              f'Loss_D: {mean_loss_d}, Loss_G: {mean_loss_g}, acc fake: {mean_acc_fake}, acc real: '
                              f'{mean_acc_real}')

                    self.logger.log_metrics(['loss_D', 'loss_G', 'acc', 'fake_acc', 'real_acc', 'acc_mean',
                                             'fake_acc_mean', 'real_acc_mean'],
                                            [loss_d.item(), loss_g.item(), acc, fake_acc.item(), real_acc.item(),
                                             np.mean(D_accs), np.mean(D_accs_fake), np.mean(D_accs_real)],
                                            self.global_step)

                    if self.global_step % 50 == 0:
                        with torch.no_grad():
                            fake = self.netG(fixed_noise).detach().cpu()
                        vutils.save_image(fake,  # torch.stack([real[0].unsqueeze((1)).cpu(), fake], 1).squeeze(0)
                                          f'../plots/epoch_{epoch}_step_{self.global_step}.png')
                    self.global_step += 1

                # save model
                self.save_model(self.netD, self.optimizerD, net='discriminator')
                self.save_model(self.netG, self.optimizerG, net='generator')

                print(f'Epoch total time: {round((time.time() - epoch_start_time) / 60, 3)} min')
            print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')

        if train_z:
            self.netD.eval()
            self.netG.eval()
            test_set_size = len(self.dl_test.dataset)
            z = torch.autograd.Variable(torch.nn.init.normal(torch.zeros(test_set_size, 100, 1, 1), mean=0, std=0.1),
                                        requires_grad=True)
            z_optimizer = torch.optim.Adam([z], lr=1e-4, betas=(0.5, 0.999))
            anomaly_scores = []
            for iter_ in range(20000):
                for i, batch in enumerate(self.dl_test):
                    images, labels, _ = batch
                    gen_fake = self.netG(z[i * self.cfg.batch_size: np.min([(i + 1) * self.cfg.batch_size,
                                                                            test_set_size])])
                    loss = self.calculate_anomaly_score(torch.autograd.Variable(images), gen_fake)
                    z_optimizer.zero_grad()
                    loss.backward()
                    z_optimizer.step()
                    anomaly_scores.append(loss.item())
                    if i % 1 == 0:
                        print(f'iter_: {iter_}, batch: {i}, loss: {loss.item()}')

                if iter_ % 100 == 0 and iter_ != 0:
                    print('Saving z...')
                    state = {
                        'z': z,
                        'epoch': iter_,
                        'global_step': self.global_step,
                        'opt': z_optimizer.state_dict()
                    }
                    if not os.path.exists(self.cfg.checkpoints_dir):
                        os.makedirs(self.cfg.checkpoints_dir)

                    path_to_save = os.path.join(self.cfg.checkpoints_dir, f'z_checkpoint_{iter_}.pth')
                    torch.save(state, path_to_save)
                    print(f'Saved z to {path_to_save}.')
        if not train_z:
            checkpoint = torch.load(self.cfg.checkpoints_dir + f'/z_checkpoint_19900.pth')
            z = checkpoint['z']
            test_set_size = len(self.dl_test.dataset)
            an_scores = []
            for i in range(test_set_size):
                image = self.dl_test.dataset.images[i].unsqueeze(0)
                gen_fake = self.netG(z[i].unsqueeze(0))
                a_scores = self.calculate_anomaly_score(torch.autograd.Variable(image), gen_fake)
                an_scores.append(a_scores.item())

            anomaly_scores = np.asarray(an_scores)
            labels = np.asarray(self.dl_test.dataset.labels)

            ids_to_sort = np.argsort(anomaly_scores)[::-1]
            predictions = anomaly_scores[ids_to_sort]
            labels = labels[ids_to_sort]

            tp = np.cumsum(labels)
            precision = tp / (np.arange(test_set_size) + 1)
            recall = tp / sum(labels != 0)

            ap_score = metrics.average_precision_score(precision, recall)
            print(f'calculated AP: {ap_score}')

            p, r, t = metrics.precision_recall_curve(labels,
                                                     predictions,
                                                     precision,
                                                     recall)

            f1_score = metrics.f1_score(p, r)
            best_f1_score_idx = np.argmax(f1_score)
            best_f1_score = f1_score[best_f1_score_idx]
            print(f'best F1-score: {best_f1_score}')

            best_thr = t[best_f1_score_idx - 1]
            fin_prediction = np.zeros(test_set_size)
            fin_prediction[predictions > best_thr] = 1

            conf_matrix_for_best_thr = metrics.confusion_matrix(labels, fin_prediction)
            print(f'Confusion matrix (tn, fp, fn, tp): {np.concatenate(conf_matrix_for_best_thr)}')
