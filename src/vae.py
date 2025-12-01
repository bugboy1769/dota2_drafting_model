import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # TODO: Define your layers here
        # 1. Linear(input_dim -> hidden_dim)
        # 2. Activation (ReLU)
        # 3. Linear(hidden_dim -> hidden_dim) (Optional, for depth)
        
        # Two separate heads for Mean and LogVariance
        # self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # self.fc_var = nn.Linear(hidden_dim, latent_dim)
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        # x -> hidden -> mu, logvar
        return None, None

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO: Define your layers here
        # Mirror the encoder
        # 1. Linear(latent_dim -> hidden_dim)
        # 2. Linear(hidden_dim -> output_dim)
        pass

    def forward(self, z):
        # TODO: Implement forward pass
        # z -> hidden -> reconstruction_logits
        return None

class DotaVAE(nn.Module):
    def __init__(self, num_heroes=150, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(num_heroes, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, num_heroes)

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick:
        z = mu + sigma * epsilon
        """
        if self.training:
            # TODO:
            # 1. Calculate std = exp(0.5 * logvar)
            # 2. Sample epsilon from standard normal (torch.randn_like)
            # 3. Return mu + std * epsilon
            pass
        else:
            # During inference, just return mu (deterministic)
            return mu

    def forward(self, x):
        # 1. Encode
        mu, logvar = self.encoder(x)
        
        # 2. Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # 3. Decode
        recon_logits = self.decoder(z)
        
        return recon_logits, mu, logvar

def vae_loss(recon_logits, x, mu, logvar):
    """
    Computes the VAE Loss = Reconstruction + KL Divergence
    """
    # 1. Reconstruction Loss (Binary Cross Entropy)
    # We use BCEWithLogitsLoss implicitly or calculate it manually
    # recon_loss = F.binary_cross_entropy_with_logits(recon_logits, x, reduction='sum')
    
    # 2. KL Divergence
    # D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return 0 # return recon_loss + kl_loss
