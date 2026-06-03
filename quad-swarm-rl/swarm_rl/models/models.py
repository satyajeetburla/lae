import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, z_dim=10, output_twice=False):
        super().__init__()
        # if output_twice, final layer has size 2*z_dim for VAE stats
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        final_dim = 2*z_dim if output_twice else z_dim
        layers.append(nn.Linear(prev, final_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, z_dim, output_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        layers = []
        prev = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

# Standard Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dims_enc=None, hidden_dims_dec=None):
        super().__init__()
        # encoder outputs z_dim
        self.encoder = Encoder(input_dim, hidden_dims_enc, z_dim, output_twice=False)
        self.decoder = Decoder(z_dim, input_dim, hidden_dims_dec)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

# VAE Base
class BaseVAE(nn.Module):
    """Vanilla VAE with MSE reconstruction and KL-divergence."""
    def __init__(self, input_dim, z_dim, hidden_dims_enc=None, hidden_dims_dec=None):
        super().__init__()
        # encoder outputs 2*z_dim (mu and logvar)
        self.z_dim = z_dim
        self.encoder = Encoder(input_dim, hidden_dims_enc, z_dim, output_twice=True)
        self.decoder = Decoder(z_dim, input_dim, hidden_dims_dec)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        stats = self.encoder(x)
        mu, logvar = stats.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return z, x_hat, mu, logvar

    def loss(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kl, {'recon_loss': recon.item(), 'kl_loss': kl.item()}

# β-VAE
class BetaVAE(BaseVAE):
    """β-VAE: scales KL term to encourage disentanglement."""
    def __init__(self, input_dim, z_dim, beta=4.0, hidden_dims_enc=None, hidden_dims_dec=None):
        super().__init__(input_dim, z_dim, hidden_dims_enc, hidden_dims_dec)
        self.beta = beta

    def loss(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss  = recon + self.beta * kl
        return loss, {
            'recon_loss': recon.item(),
            'kl_loss': kl.item(),
            'beta': self.beta
        }

# FactorVAE
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Discriminator(nn.Module):
#     """Discriminator for FactorVAE TC estimation, with logit clamping."""
#     def __init__(self, z_dim, hidden_dims=None, logit_clip=20.0):
#         super().__init__()
#         if hidden_dims is None:
#             hidden_dims = [1000, 1000]
#         layers = []
#         prev = z_dim
#         for h in hidden_dims:
#             layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
#             prev = h
#         layers.append(nn.Linear(prev, 2))  # [marginal, joint]
#         self.net = nn.Sequential(*layers)
#         self.logit_clip = logit_clip
#
#     def forward(self, z):
#         logits = self.net(z)
#         # clamp to avoid extreme values
#         return logits.clamp(min=-self.logit_clip, max=self.logit_clip)

class Discriminator(nn.Module):
    """Discriminator for FactorVAE TC estimation, with logit clamping."""
    def __init__(self, z_dim, hidden_dims=None, logit_clip=20.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1000, 1000]
        layers = []
        prev = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 2))  # [marginal, joint]
        self.net = nn.Sequential(*layers)
        self.logit_clip = logit_clip

    def forward(self, z):
        logits = self.net(z)
        # clamp to avoid extreme values
        return logits.clamp(min=-self.logit_clip, max=self.logit_clip)

#
# class FactorVAE(BetaVAE):
#     """FactorVAE: β-VAE + total-correlation penalty via discriminator."""
#     def __init__(self, input_dim, z_dim, beta=4.0, tc_weight=6.0,
#                  hidden_dims_enc=None, hidden_dims_dec=None,
#                  disc_hidden_dims=None, logit_clip=20.0):
#         super().__init__(input_dim, z_dim, beta, hidden_dims_enc, hidden_dims_dec)
#         self.tc_weight = tc_weight
#         self.discriminator = Discriminator(z_dim, hidden_dims=disc_hidden_dims, logit_clip=logit_clip)
#
#     def permute_dims(self, z):
#         # Permute each latent dimension independently
#         z_perm = []
#         batch_size = z.size(0)
#         for i in range(z.size(1)):
#             z_perm.append(z[:, i][torch.randperm(batch_size)])
#         return torch.stack(z_perm, dim=1)
#
#     def tc_loss(self, z):
#         # joint vs marginal logits
#         joint_logits = self.discriminator(z)
#         z_perm = self.permute_dims(z)
#         marg_logits = self.discriminator(z_perm)
#
#         # density-ratio estimate: log q(z)/prod q(z_j)
#         tc_term = joint_logits[:, 1] - marg_logits[:, 1]
#
#         # clip gradients on the TC term to avoid explosion
#         tc_term = torch.clamp(tc_term, min=-10.0, max=10.0)
#         tc = tc_term.mean()
#
#         # guard against NaNs
#         if torch.isnan(tc):
#             print("[FactorVAE] ⚠️  NaN in tc_loss(): joint/marg logits:",
#                   joint_logits[0], marg_logits[0])
#             tc = torch.zeros_like(tc)
#
#         return tc
#
#     def loss(self, x, x_hat, mu, logvar, z=None):
#         if z is None:
#             z, x_hat, mu, logvar = self.forward(x)
#
#         recon = F.mse_loss(x_hat, x, reduction='mean')
#         kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#
#         tc = self.tc_loss(z)
#         loss = recon + self.beta * kl + self.tc_weight * tc
#
#         return loss, {
#             'recon_loss': recon.item(),
#             'kl_loss': kl.item(),
#             'tc_loss': tc.item(),
#             'beta': self.beta,
#             'tc_weight': self.tc_weight
#
#         }


class FactorVAE(BetaVAE):
    """FactorVAE: β-VAE + total-correlation penalty via discriminator."""
    def __init__(self, input_dim, z_dim, beta=4.0, tc_weight=6.0,
                 hidden_dims_enc=None, hidden_dims_dec=None,
                 disc_hidden_dims=None, logit_clip=20.0):
        super().__init__(input_dim, z_dim, beta, hidden_dims_enc, hidden_dims_dec)
        self.tc_weight = tc_weight
        self.discriminator = Discriminator(z_dim, hidden_dims=disc_hidden_dims, logit_clip=logit_clip)

    def permute_dims(self, z):
        # Permute each latent dimension independently
        z_perm = []
        batch_size = z.size(0)
        for i in range(z.size(1)):
            z_perm.append(z[:, i][torch.randperm(batch_size)])
        return torch.stack(z_perm, dim=1)

    def tc_loss(self, z):
        # joint vs marginal logits
        joint_logits = self.discriminator(z)
        z_perm = self.permute_dims(z)
        marg_logits = self.discriminator(z_perm)

        # density-ratio estimate: log q(z)/prod q(z_j)
        tc_term = joint_logits[:, 1] - marg_logits[:, 1]

        # clip gradients on the TC term to avoid explosion
        tc_term = torch.clamp(tc_term, min=-10.0, max=10.0)
        tc = tc_term.mean()

        # guard against NaNs
        if torch.isnan(tc):
            print("[FactorVAE] ⚠️  NaN in tc_loss(): joint/marg logits:",
                  joint_logits[0], marg_logits[0])
            tc = torch.zeros_like(tc)

        return tc

    def loss(self, x, x_hat, mu, logvar, z=None):
        if z is None:
            z, x_hat, mu, logvar = self.forward(x)

        recon = F.mse_loss(x_hat, x, reduction='mean')
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        tc = self.tc_loss(z)
        loss = recon + self.beta * kl + self.tc_weight * tc

        return loss, {
            'recon_loss': recon.item(),
            'kl_loss': kl.item(),
            'tc_loss': tc.item(),
            'beta': self.beta,
            'tc_weight': self.tc_weight
        }


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dims=None, z_dim=10, output_twice=False):
#         super().__init__()
#         # if output_twice, final layer has size 2*z_dim for VAE stats
#         if hidden_dims is None:
#             hidden_dims = [128, 64]
#         layers = []
#         prev = input_dim
#         for h in hidden_dims:
#             layers += [nn.Linear(prev, h), nn.ReLU()]
#             prev = h
#         final_dim = 2*z_dim if output_twice else z_dim
#         layers.append(nn.Linear(prev, final_dim))
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.net(x)
#
# class Decoder(nn.Module):
#     def __init__(self, z_dim, output_dim, hidden_dims=None):
#         super().__init__()
#         if hidden_dims is None:
#             hidden_dims = [64, 128]
#         layers = []
#         prev = z_dim
#         for h in hidden_dims:
#             layers += [nn.Linear(prev, h), nn.ReLU()]
#             prev = h
#         layers.append(nn.Linear(prev, output_dim))
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, z):
#         return self.net(z)
#
# class Encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dims=None, z_dim=10, output_twice=False):
#         super().__init__()
#         # if output_twice, final layer has size 2*z_dim for VAE stats
#         if hidden_dims is None:
#             hidden_dims = [64, 128, 256, 512, 256, 128, 64]
#         layers = []
#         prev = input_dim
#         for h in hidden_dims:
#             layers += [nn.Linear(prev, h), nn.ReLU()]
#             prev = h
#         final_dim = 2*z_dim if output_twice else z_dim
#         layers.append(nn.Linear(prev, final_dim))
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.net(x)
#
# class Decoder(nn.Module):
#     def __init__(self, z_dim, output_dim, hidden_dims=None):
#         super().__init__()
#         if hidden_dims is None:
#             hidden_dims = [ 64, 128, 256,512,256, 128, 64]
#         layers = []
#         prev = z_dim
#         for h in hidden_dims:
#             layers += [nn.Linear(prev, h), nn.ReLU()]
#             prev = h
#         layers.append(nn.Linear(prev, output_dim))
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, z):
#         return self.net(z)
#
#
# # Standard Autoencoder
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, z_dim, hidden_dims_enc=None, hidden_dims_dec=None):
#         super().__init__()
#         # encoder outputs z_dim
#         self.encoder = Encoder(input_dim, hidden_dims_enc, z_dim, output_twice=False)
#         self.decoder = Decoder(z_dim, input_dim, hidden_dims_dec)
#
#     def forward(self, x):
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         return z, x_hat
#
# # VAE Base
# class BaseVAE(nn.Module):
#     """Vanilla VAE with MSE reconstruction and KL-divergence."""
#     def __init__(self, input_dim, z_dim, hidden_dims_enc=None, hidden_dims_dec=None):
#         super().__init__()
#         # encoder outputs 2*z_dim (mu and logvar)
#         self.z_dim = z_dim
#         self.encoder = Encoder(input_dim, hidden_dims_enc, z_dim, output_twice=True)
#         self.decoder = Decoder(z_dim, input_dim, hidden_dims_dec)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         stats = self.encoder(x)
#         mu, logvar = stats.chunk(2, dim=1)
#         z = self.reparameterize(mu, logvar)
#         x_hat = self.decoder(z)
#         return z, x_hat, mu, logvar
#
#     def loss(self, x, x_hat, mu, logvar):
#         recon = F.mse_loss(x_hat, x, reduction='mean')
#         kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#         return recon + kl, {'recon_loss': recon.item(), 'kl_loss': kl.item()}
#
# # β-VAE
# class BetaVAE(BaseVAE):
#     """β-VAE: scales KL term to encourage disentanglement."""
#     def __init__(self, input_dim, z_dim, beta=4.0, hidden_dims_enc=None, hidden_dims_dec=None):
#         super().__init__(input_dim, z_dim, hidden_dims_enc, hidden_dims_dec)
#         self.beta = beta
#
#     def loss(self, x, x_hat, mu, logvar):
#         recon = F.mse_loss(x_hat, x, reduction='mean')
#         kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#         loss  = recon + self.beta * kl
#         return loss, {
#             'recon_loss': recon.item(),
#             'kl_loss': kl.item(),
#             'beta': self.beta
#         }
#
# # FactorVAE
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Discriminator(nn.Module):
#     """Discriminator for FactorVAE TC estimation, with logit clamping."""
#     def __init__(self, z_dim, hidden_dims=None, logit_clip=20.0):
#         super().__init__()
#         if hidden_dims is None:
#             hidden_dims = [512, 1024, 512]
#         layers = []
#         prev = z_dim
#         for h in hidden_dims:
#             layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
#             prev = h
#         layers.append(nn.Linear(prev, 2))  # [marginal, joint]
#         self.net = nn.Sequential(*layers)
#         self.logit_clip = logit_clip
#
#     def forward(self, z):
#         logits = self.net(z)
#         # clamp to avoid extreme values
#         return logits.clamp(min=-self.logit_clip, max=self.logit_clip)
#
#
# class FactorVAE(BetaVAE):
#     """FactorVAE: β-VAE + total-correlation penalty via discriminator."""
#     def __init__(self, input_dim, z_dim, beta=4.0, tc_weight=6.0,
#                  hidden_dims_enc=None, hidden_dims_dec=None,
#                  disc_hidden_dims=None, logit_clip=20.0):
#         super().__init__(input_dim, z_dim, beta, hidden_dims_enc, hidden_dims_dec)
#         self.tc_weight = tc_weight
#         self.discriminator = Discriminator(z_dim, hidden_dims=disc_hidden_dims, logit_clip=logit_clip)
#
#     def permute_dims(self, z):
#         # Permute each latent dimension independently
#         z_perm = []
#         batch_size = z.size(0)
#         for i in range(z.size(1)):
#             z_perm.append(z[:, i][torch.randperm(batch_size)])
#         return torch.stack(z_perm, dim=1)
#
#     def tc_loss(self, z):
#         # joint vs marginal logits
#         joint_logits = self.discriminator(z)
#         z_perm = self.permute_dims(z)
#         marg_logits = self.discriminator(z_perm)
#
#         # density-ratio estimate: log q(z)/prod q(z_j)
#         tc_term = joint_logits[:, 1] - marg_logits[:, 1]
#
#         # clip gradients on the TC term to avoid explosion
#         tc_term = torch.clamp(tc_term, min=-10.0, max=10.0)
#         tc = tc_term.mean()
#
#         # guard against NaNs
#         if torch.isnan(tc):
#             print("[FactorVAE] ⚠️  NaN in tc_loss(): joint/marg logits:",
#                   joint_logits[0], marg_logits[0])
#             tc = torch.zeros_like(tc)
#
#         return tc
#
#     def loss(self, x, x_hat, mu, logvar, z=None):
#         if z is None:
#             z, x_hat, mu, logvar = self.forward(x)
#
#         recon = F.mse_loss(x_hat, x, reduction='mean')
#         kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#
#         tc = self.tc_loss(z)
#         loss = recon + self.beta * kl + self.tc_weight * tc
#
#         return loss, {
#             'recon_loss': recon.item(),
#             'kl_loss': kl.item(),
#             'tc_loss': tc.item(),
#             'beta': self.beta,
#             'tc_weight': self.tc_weight
#         }