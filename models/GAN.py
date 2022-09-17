from torch import nn


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.cfg = cfg

        self.conv1 = nn.ConvTranspose2d(self.cfg.nz, self.cfg.ngf * 8, 4, 1, 0, bias=False)
        self.norm1 = LayerNorm2d(self.cfg.ngf * 8) if self.cfg.use_layer_norm else nn.BatchNorm2d(self.cfg.ngf * 8)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(self.cfg.ngf * 8, self.cfg.ngf * 4, 4, 2, 1, bias=False)
        self.norm2 = LayerNorm2d(self.cfg.ngf * 4) if self.cfg.use_layer_norm else nn.BatchNorm2d(self.cfg.ngf * 4)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.ConvTranspose2d(self.cfg.ngf * 4, self.cfg.ngf * 2, 4, 2, 1, bias=False)
        self.norm3 = LayerNorm2d(self.cfg.ngf * 2) if self.cfg.use_layer_norm else nn.BatchNorm2d(self.cfg.ngf * 2)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = nn.ConvTranspose2d(self.cfg.ngf * 2, self.cfg.ngf, 4, 2, 1, bias=False)
        self.norm4 = LayerNorm2d(self.cfg.ngf) if self.cfg.use_layer_norm else nn.BatchNorm2d(self.cfg.ngf)
        self.act4 = nn.ReLU(inplace=True)
        self.conv5 = nn.ConvTranspose2d(self.cfg.ngf, self.cfg.nc, 4, 2, 1, bias=False)
        self.act5 = nn.Tanh()

        self.generator = nn.Sequential(self.conv1, self.norm1, self.act1,
                                       self.conv2, self.norm2, self.act2,
                                       self.conv3, self.norm3, self.act3,
                                       self.conv4, self.norm4, self.act4,
                                       self.conv5, self.act5)

    def forward(self, x):
        x = self.generator(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.cfg = cfg

        self.conv1 = nn.Conv2d(self.cfg.nc, self.cfg.ndf, 4, 2, 1, bias=False)
        self.act1 = nn.LeakyReLU(self.cfg.leaky_relu_param, inplace=True)
        self.conv2 = nn.Conv2d(self.cfg.ndf, self.cfg.ndf * 2, 4, 2, 1, bias=False)
        self.norm2 = LayerNorm2d(self.cfg.ngf * 2) if self.cfg.use_layer_norm else nn.BatchNorm2d(self.cfg.ndf * 2)
        self.act2 = nn.LeakyReLU(self.cfg.leaky_relu_param, inplace=True)
        self.conv3 = nn.Conv2d(self.cfg.ndf * 2, self.cfg.ndf * 4, 4, 2, 1, bias=False)
        self.norm3 = LayerNorm2d(self.cfg.ngf * 4) if self.cfg.use_layer_norm else nn.BatchNorm2d(self.cfg.ndf * 4)
        self.act3 = nn.LeakyReLU(self.cfg.leaky_relu_param, inplace=True)
        self.conv4 = nn.Conv2d(self.cfg.ndf * 4, self.cfg.ndf * 8, 4, 2, 1, bias=False)
        self.norm4 = LayerNorm2d(self.cfg.ngf * 8) if self.cfg.use_layer_norm else nn.BatchNorm2d(self.cfg.ndf * 8)
        self.act4 = nn.LeakyReLU(self.cfg.leaky_relu_param, inplace=True)
        self.conv5 = nn.Conv2d(self.cfg.ndf * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.discriminator = nn.Sequential(self.conv1, self.act1,
                                           self.conv2, self.norm2, self.act2,
                                           self.conv3, self.norm3, self.act3,
                                           self.conv4, self.norm4, self.act4,
                                           self.conv5)

    def forward(self, x):
        out = self.discriminator(x)
        feature = out.view(out.size()[0], -1)
        x = self.sigmoid(out)
        return x, feature


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, affine: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels, num_channels, affine=affine)

    def forward(self, x):
        return self.norm(x)


def get_model(cfg):
    """
    Gets model.
    """
    generator = Generator(cfg)
    if cfg.device == 'cuda':
        generator = generator.cuda()
    generator.apply(weights_init)
    print(generator)

    discriminator = Discriminator(cfg)
    if cfg.device == 'cuda':
        discriminator = discriminator.cuda()
    discriminator.apply(weights_init)
    print(discriminator)

    return generator, discriminator
