# AnoGAN

## About The Project

1) Implementation of AnoGAN architecture from `Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery` paper:

        https://arxiv.org/pdf/1703.05921.pdf

2) Training on MVTec AD (MVTec Anomaly Detection Dataset):

        https://www.mvtec.com/company/research/datasets/mvtec-ad

for anomaly detection task.


## Getting Started

File to run:

    /executor/executor.py 
    
- After running executor.py, checkpoints for GANs will be restored from `/saved_files/checkpoints/`;
- Checkpoint for latent vectors will also be loaded from `/saved_files/checkpoints/` (was learning for 20k iterations).
- Calculation of AP and F1 metrics, as well as building the confusion matrix for loaded checkpoints are in the train function in the `trainers/trainer.py` file.

Visualization of loss for the discriminator and generator, as well as examples of generated images are in:

    /plots/
    
    
