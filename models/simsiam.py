import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    elif version == 'symmetric':
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}

class SimSiamKL(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        
        self.projector_mu = projection_MLP(in_dim=proj_dim, out_dim=proj_dim)
        self.projector_var = projection_MLP(in_dim=proj_dim, out_dim=proj_dim)

    def forward(self, x1, x2, sym_loss_weight=1.0, logistic_loss_weight=0.0):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # Batch and output dimensionality
        B, N = z1.shape
        device = z1.device

        z1_mu, z1_var = self.projector_mu(z1), torch.sigmoid(self.projector_var(z1))
        z2_mu, z2_var = self.projector_mu(z2), torch.sigmoid(self.projector_var(z2))

        z1_mu_norm = (z1_mu - z1_mu.mean(0)) / z1_mu.std(0)
        z2_mu_norm = (z2_mu - z2_mu.mean(0)) / z2_mu.std(0)
        
        # z1_logvar = torch.log(z1_var)
        # z2_logvar = torch.log(z2_var)

        # z1_mu, z1_tril_vec = self.projector_mu(z1), self.projector_var(z1)
        # z2_mu, z2_tril_vec = self.projector_mu(z2), self.projector_var(z2)

        # # Figure this out
        # # Convert lower triangular matrix in vector form to matrix form
        # tril_indices = torch.tril_indices(row=N, col=N, offset=0)
        # z1_tril_mat = z2_tril_mat = torch.zeros((B, N, N), device=device)
        # z1_tril_mat[:, tril_indices[0], tril_indices[1]] = z1_tril_vec
        # z2_tril_mat[:, tril_indices[0], tril_indices[1]] = z2_tril_vec

        # # Soft-plusing diagonal elements (Need to clone b/c of in-place operations and need to preserve original items)
        # z1_tril_mat_c = z1_tril_mat.clone()
        # z2_tril_mat_c = z2_tril_mat.clone()
        # z1_tril_mat_c[:, range(N), range(N)] = F.softplus(z1_tril_mat.diagonal(dim1=-2, dim2=-1))
        # z2_tril_mat_c[:, range(N), range(N)] = F.softplus(z2_tril_mat.diagonal(dim1=-2, dim2=-1))

        # Pytorch internal method of KL
        z1_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=z1_mu, covariance_matrix=torch.diag(z1_var))
        z2_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=z2_mu, scale_tril=torch.diag(z2_var))

        z1_kl = torch.distributions.kl.kl_divergence(z1_dist, z2_dist.detach())
        z2_kl = torch.distributions.kl.kl_divergence(z2_dist, z1_dist.detach())

        # gaus_dist2 = torch.distributions.multivariate_normal.MultivariateNormal(loc=z2_mu, scale_tril=torch.diag(torch.ones(N, device=device)))
        # z2_kl = torch.distributions.kl.kl_divergence(z2_dist, gaus_dist2)

        # Exact numerical calculation of KL

        # Get covariance matrix from L => L * L^transpose (exclude batch dimension in transpose)
        # z1_cov = torch.matmul(z1_var_mat, torch.transpose(z1_var_mat, 1, 2))
        # z2_cov = torch.matmul(z2_var_mat, torch.transpose(z2_var_mat, 1, 2))

        # z1_kl = 0.5 * torch.sum(-torch.logdet(z1_cov) - N + torch.trace(z1_cov) + torch.matmul((z2_mu - z1_mu).T, (z2_mu - z1_mu)))
        # z2_kl = 0.5 * torch.sum(-torch.logdet(z2_cov) - N + torch.trace(z2_cov) + torch.matmul((z1_mu - z2_mu).T, (z1_mu - z2_mu)))

        # Calculate KL divergence between z1, z2, gaussian with the same mean
        # z1_kl = -0.5 * torch.sum(1 + z1_logvar - z1_logvar.exp(), dim=-1).mean()
        # z2_kl = -0.5 * torch.sum(1 + z2_logvar - z2_logvar.exp(), dim=-1).mean()

        # Reparameterize with same eps
        # z1, z2 = self.reparameterize(z1_mu_norm, z1_logvar, z2_mu_norm, z2_logvar)
        # z2 = self.reparameterize(z2_mu, z2_logvar)

        # Reparameterize
        # z1 = z1_dist.rsample()
        # z2 = z2_dist.rsample()

        loss_kl = z1_kl * 0.5 + z2_kl * 0.5
        # loss_pos = - F.cosine_similarity(z1, z2)
        # loss_simclr = NT_XentLoss(z1, z2)
        # loss_pos = gaussian_kernel_pos_loss(z1_mu, z2_mu)
        
        loss = loss_kl
        # + z1.shape[0] * loss_simclr
        return {'loss': loss, 'loss/kl': loss_kl}
        # return {'loss': loss, 'loss/pos': loss_pos, 'loss/kl': loss_kl}

    # def reparameterize(self, mu1, logvar1, mu2, logvar2):
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std1 = torch.exp(0.5 * logvar1)
    #     std2 = torch.exp(0.5 * logvar2)
    #     eps1 = torch.randn_like(std1)
    #     eps2 = torch.randn_like(std2)
    #     return eps1 * std1 + mu1, eps2 * std2 + mu2

############################################################################
# Knowledge Distillation Models
############################################################################

class SimSiamKD(nn.Module):
    def __init__(self, backbones):
        super().__init__()
        
        self.backbone_s, self.backbone_t = backbones
        self.projector = projection_MLP(self.backbone_s.output_dim)

        # Student encoder
        self.encoder_s = nn.Sequential( # f encoder
            self.backbone_s,
            self.projector
        )

        # Teacher encoder
        self.encoder_t =  nn.Sequential( # f encoder
            self.backbone_t,
            # self.projector
        )
        self.predictor = prediction_MLP(in_dim=self.projector.out_dim, out_dim=self.backbone_t.output_dim)
    
    def forward(self, x1, x2):

        f_s, f_t, h = self.encoder_s, self.encoder_t, self.predictor
        z1_1, z1_2, z2_1, z2_2 = f_s(x1), f_s(x2), f_t(x1), f_t(x2)
        p1, p2 = h(z1_1), h(z1_2)
        L = D(p1, z2_1) / 2 + D(p2, z2_2) / 2
        return {'loss': L}

class SimSiamKDAnchor(nn.Module):
    def __init__(self, backbones=[resnet50, resnet50]):
        super().__init__()
        
        self.backbone_s, self.backbone_t = backbones
        self.projector = projection_MLP(in_dim=self.backbone_s.output_dim)

        # Student encoder
        self.encoder_s = nn.Sequential( # f encoder
            self.backbone_s,
            self.projector
        )

        # Teacher encoder
        self.encoder_t =  nn.Sequential( # f encoder
            self.backbone_t,
            # self.projector
        )
        self.predictor = prediction_MLP(in_dim=self.projector.out_dim, out_dim=self.backbone_t.output_dim)
    
    def forward(self, x1, x2, x3):

        f_s, f_t, h = self.encoder_s, self.encoder_t, self.predictor
        z1_1, z1_2, z2 = f_s(x1), f_s(x2), f_t(x3)
        p1, p2 = h(z1_1), h(z1_2)
        L = D(p1, z2) / 2 + D(p2, z2) / 2
        return {'loss': L}

############################################################################
# Adversarial Formulation
############################################################################
class Discriminator(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512):
        super().__init__()
        ''' 
        Discriminator for estimating ratio (joint / marginal)
        '''

        self.in_dim = in_dim
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)

        return x 

class SimSiamAdv(nn.Module):
    def __init__(self, backbone=resnet50(), proj_dim=128):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(in_dim=backbone.output_dim, out_dim=proj_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )

        self.discriminator = Discriminator(in_dim=proj_dim*2)

    def forward(self, x1, x2, disc=False):
        if not disc:
            return self.forward_e(x1, x2)
        else:
            return self.forward_d(x1, x2)
        
    def forward_e(self, x1, x2):

        f, d = self.encoder, self.discriminator
        z1, z2 = f(x1), f(x2)
        L = -1.0 * d(torch.cat((z1, z2), dim=-1))
        
        return {'loss_e': L}
    
    def forward_d(self, x1, x2):
        f, d = self.encoder, self.discriminator
        z1, z2 = f(x1), f(x2)
        
        real = torch.ones((x1.shape[0], 1), dtype=torch.float32, device=x1.device)
        fake = torch.zeros((x1.shape[0], 1), dtype=torch.float32, device=x1.device)

        real_outputs = d(torch.cat((z1, z2), dim=-1))
        fake_outputs = d(torch.cat((z1[torch.randperm(z1.size()[0])], z2[torch.randperm(z2.size()[0])]), dim=-1))
        
        real_loss = F.binary_cross_entropy(real_outputs, real)
        fake_loss = F.binary_cross_entropy(fake_outputs, fake)

        return {'loss_d': (real_loss + fake_loss) / 2, 'loss_d_real': real_loss, 'loss_d_fake': fake_loss}


class SimSiamJoint(nn.Module):
    def __init__(self, backbone=resnet50(), proj_dim=128):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(in_dim=backbone.output_dim, out_dim=proj_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )

        self.discriminator = Discriminator(in_dim=proj_dim*2)
    
    def forward(self, x1, x2, sym_loss_weight=1.0, logistic_loss_weight=1.0):
        f, d = self.encoder, self.discriminator
        z1, z2 = f(x1), f(x2)

        sym_loss = D(z1, z2, version='symmetric') if sym_loss_weight > 0.0 else 0.0
        
        if logistic_loss_weight > 0.0:
            real = torch.ones((x1.shape[0], 1), dtype=torch.float32, device=x1.device)
            fake = torch.zeros((x1.shape[0], 1), dtype=torch.float32, device=x1.device)

            real_outputs = d(torch.cat((z1, z2), dim=-1))
            fake_outputs = d(torch.cat((z1[torch.randperm(z1.size()[0])], z2[torch.randperm(z2.size()[0])]), dim=-1))
            
            real_loss = F.binary_cross_entropy(real_outputs, real)
            fake_loss = F.binary_cross_entropy(fake_outputs, fake)

            d_loss = ((real_loss + fake_loss) / 2 * logistic_loss_weight) if logistic_loss_weight > 0.0 else 0.0

        # No symmetric loss
        if sym_loss_weight <= 0.0:
            return {'loss': d_loss, 'loss_d': d_loss, 'loss_d_real': real_loss, 'loss_d_fake': fake_loss}
        # No logistic loss
        elif logistic_loss_weight <= 0.0:
            return {'loss': sym_loss, 'loss_sym': sym_loss}
        # Both symmetric and logistic loss present
        else:
            return {'loss': sym_loss + d_loss, 'loss_sym': sym_loss, 'loss_d': d_loss, 'loss_d_real': real_loss, 'loss_d_fake': fake_loss}

# Original SimSiam + Discriminator at z level
class SimSiamMI(nn.Module):
    def __init__(self, backbone=resnet50(), proj_dim=2048):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim, out_dim=proj_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

        self.discriminator = Discriminator(in_dim=proj_dim*2)
    
    def forward(self, x1, x2, disc=False):
        if disc:
            return self.forward_d(x1, x2)
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L, 'loss_sym': L}
    
    def forward_d(self, x1, x2):
        f, d = self.encoder, self.discriminator
        z1, z2 = f(x1), f(x2)
        
        real = torch.ones((x1.shape[0], 1), dtype=torch.float32, device=x1.device)
        fake = torch.zeros((x1.shape[0], 1), dtype=torch.float32, device=x1.device)

        real_outputs = d(torch.cat((z1, z2), dim=-1))
        fake_outputs = d(torch.cat((z1[torch.randperm(z1.size()[0])], z2[torch.randperm(z2.size()[0])]), dim=-1))
        
        real_loss = F.binary_cross_entropy(real_outputs, real)
        fake_loss = F.binary_cross_entropy(fake_outputs, fake)

        d_loss = (real_loss + fake_loss) / 2

        mi = -1.0 * real_outputs
        
        return {'loss_d/total': d_loss, 'loss_d/real': real_loss, 'loss_d/fake': fake_loss, 'loss_d/mi': mi}

if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












